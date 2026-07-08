import os
import time
import argparse
from pathlib import Path
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

# Utilidades locales del proyecto.
from loss_func import mse_loss, bce_loss
from data_load import StarDataSet, star_collate_fn, compute_mean_std

# Configuración de optimización de PyTorch.
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Inicialización de Mixed Precision para GPUs NVIDIA.
scaler = GradScaler(enabled=torch.cuda.is_available())

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DATA_DIR = PROJECT_ROOT / "data_generation" / "training_data"
SAVED_MODELS_DIR = SCRIPT_DIR / "saved_models"
RUNS_DIR = SCRIPT_DIR / "runs"

def validation(model, criterion_bce, val_dataloader, device):
    """Evalúa el modelo en el conjunto de validación con precisión mixta."""
    model.eval()
    running_dist_loss = 0.0 
    running_seg_loss = 0.0
    amp_enabled = (device.type == 'cuda')

    with torch.no_grad():
        for batch in val_dataloader:
            images, dist_map, seg_map, _ = batch
            images = images.to(device, non_blocking=True)
            dist_map = dist_map.to(device, non_blocking=True)
            seg_map = seg_map.to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = model(images)
                seg_prediction = prediction[:, 0]
                dist_prediction = prediction[:, 1]

                # Cálculo de pérdida balanceada (Masked MSE).
                active_pixels = seg_map.sum() + 1e-8
                loss_dist = torch.sum(((dist_prediction - dist_map) * seg_map) ** 2) / active_pixels
                loss_seg = criterion_bce(seg_prediction, seg_map)

            running_dist_loss += loss_dist.item()
            running_seg_loss += loss_seg.item()

    # Promedio de pérdidas
    running_dist_loss /= len(val_dataloader)
    running_seg_loss /= len(val_dataloader)

    return running_dist_loss, running_seg_loss

def training(model, criterion_mse, criterion_bce, optimizer, train_dataloader, val_dataloader, epoch, writer_dict, device):
    """Ciclo de entrenamiento principal con balanceo de gradientes."""
    model.train()
    running_dist_loss = 0.0
    running_seg_loss = 0.0
    amp_enabled = (device.type == 'cuda')
    start_time = time.time()

    for batch in train_dataloader:
        images, dist_map, seg_map, _ = batch
        images = images.to(device, non_blocking=True)
        dist_map = dist_map.to(device, non_blocking=True)
        seg_map = seg_map.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            prediction = model(images)
            seg_prediction = prediction[:, 0]
            dist_prediction = prediction[:, 1]

            # Balanceo de gradientes para mejorar la deteccion en nebulosas.
            # Se usa un peso de 0.001 para que la distancia no domine sobre la segmentación.
            active_pixels = seg_map.sum() + 1e-8
            loss_dist = torch.sum(((dist_prediction - dist_map) * seg_map) ** 2) / active_pixels
            loss_seg = criterion_bce(seg_prediction, seg_map)
            
            # Pérdida total combinada.
            loss = 0.001 * loss_dist + loss_seg

        # Backpropagation con escalado de gradientes (Mixed Precision).
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_dist_loss += loss_dist.item()
        running_seg_loss += loss_seg.item()

    # Normalización y ejecución de validación.
    running_dist_loss /= len(train_dataloader)
    running_seg_loss /= len(train_dataloader)

    with torch.no_grad():
        val_dist, val_seg = validation(model, criterion_mse, criterion_bce, val_dataloader, device)

    # Reporte de progreso en consola.
    print(f'Epoch {epoch:03d} | Train Dist: {running_dist_loss:.5f} | Seg: {running_seg_loss:.5f} | Val Dist: {val_dist:.5f} | Seg: {val_seg:.5f} | LR: {optimizer.param_groups[0]["lr"]:.2e} | {time.time() - start_time:.1f}s')

    # Almacenar métricas para logs.
    writer_dict['epoch'].append(epoch)
    writer_dict['train_dist'].append(running_dist_loss)
    writer_dict['train_seg'].append(running_seg_loss)
    writer_dict['val_dist'].append(val_dist)
    writer_dict['val_seg'].append(val_seg)

    return writer_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetuning Controlado V2 - Star Tracker")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w", type=float, default=5)  # Weight decay.
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--ep", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--create", type=int, default=0)  # 0: Cargar, 1: Crear.
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--patch_h", type=int, default=480)
    parser.add_argument("--patch_w", type=int, default=480)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--compute_stats", type=int, default=0)

    args = parser.parse_args()

    # Selección automática de dispositivo (GPU/CPU).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Inicialización de logs de TensorBoard.
    writer_tb = SummaryWriter(log_dir=str(RUNS_DIR / f"finetune_trial_{args.trial}"))
    torch.manual_seed(args.seed)

    # Carga de arquitectura o modelo existente.
    if args.create == 1:
        from neural_net.elunet import ELUnet
        model = ELUnet(1, 2, 8).to(device)
        print(f"Nuevo modelo {model.__class__.__name__} creado.")
    else:
        if args.load is None:
            raise ValueError("Debe proporcionar un modelo con --load para el finetuning.")
        load_path = Path(args.load)
        if not load_path.is_absolute():
            load_path = (SCRIPT_DIR / load_path).resolve()
        model = torch.load(load_path, map_location=device, weights_only=False).to(device)
        print(f"Modelo cargado desde: {load_path}")

    writer_info = {'epoch': [], 'train_dist': [], 'train_seg': [], 'val_dist': [], 'val_seg': []}

    # Configuración de normalización (Métricas de Stellarium V2).
    mean_val = [5.9355394]
    std_val = [12.10830327]
    use_pin = bool(args.pin_memory) and (device.type == 'cuda')
    patch_size = (args.patch_h, args.patch_w)

    if args.compute_stats == 1:
        print("Calculando nuevas estadísticas del dataset...")
        stats_loader = data.DataLoader(StarDataSet(split='train', norm=False, data_dir=TRAINING_DATA_DIR), batch_size=1, num_workers=4)
        mean_val, std_val = compute_mean_std(stats_loader)
        mean_val, std_val = mean_val.tolist(), std_val.tolist()

    # Preparación de DataLoaders con Tiling/Random Crop.
    train_loader = data.DataLoader(
        StarDataSet(split='train', data_dir=TRAINING_DATA_DIR, norm=True, mean=mean_val, std=std_val, random_crop=True, patch_size=patch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=use_pin, collate_fn=star_collate_fn
    )
    val_loader = data.DataLoader(
        StarDataSet(split="val", data_dir=TRAINING_DATA_DIR, norm=True, mean=mean_val, std=std_val, random_crop=True, patch_size=patch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=use_pin, collate_fn=star_collate_fn
    )

    # Configuración de optimización.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w * 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Pérdida balanceada: ponderación de 99 para compensar el desbalance de píxeles.
    criterion_mse = mse_loss('mean')
    pos_weight = torch.tensor([99.0]).to(device)
    criterion_bce = bce_loss(pos_weight=pos_weight)

    # Bucle de entrenamiento principal.
    for epoch in range(args.ep):
        writer_info = training(model, criterion_mse, criterion_bce, optimizer, train_loader, val_loader, epoch, writer_info, device)
        scheduler.step()

        # Registro en TensorBoard.
        writer_tb.add_scalar('train_dist', writer_info['train_dist'][-1], epoch)
        writer_tb.add_scalar('train_seg', writer_info['train_seg'][-1], epoch)
        writer_tb.add_scalar('val_dist', writer_info['val_dist'][-1], epoch)
        writer_tb.add_scalar('val_seg', writer_info['val_seg'][-1], epoch)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = SAVED_MODELS_DIR / f"{model.__class__.__name__}_{args.trial}_{epoch+1}.pt"
            torch.save(model, checkpoint_path)
            print(f"Checkpoint guardado en época {epoch+1}")

    print("Entrenamiento finalizado exitosamente.")
