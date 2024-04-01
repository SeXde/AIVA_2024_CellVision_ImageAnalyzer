import os
import time
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from train_constants import (
    DEVICE,
    NUM_CLASSES,
    CLASSES,
    BATCH_SIZE,
    NUM_EPOCHS,
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES,
    NUM_WORKERS,
    RESIZE_TO,
    TRAIN,
    VALID
)
from utils.model_utils import create_fcos_model
from train_utils import Averager, SaveBestModel, save_model, save_loss_plot, save_mAP
from datasets import create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

plt.style.use('ggplot')
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Function for running training iterations.
def train(train_data_loader, model, optimizer, train_loss_hist):
    print('Training')
    model.train()
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value


# Function for running validation iterations.
def validate(valid_data_loader, model, metric):
    print('Validating')
    model.eval()
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target, preds = [], []
    for _, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            outputs = model(images, targets)
        for i in range(len(images)):
            true_dict, preds_dict = dict(), dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
    metric.reset()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    train_dataset = create_train_dataset(TRAIN, CLASSES, RESIZE_TO)
    valid_dataset = create_valid_dataset(VALID, CLASSES, RESIZE_TO)
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    print(f"Using device: {DEVICE}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    model = create_fcos_model(num_classes=NUM_CLASSES, min_size=RESIZE_TO, max_size=RESIZE_TO).to(DEVICE)
    if os.path.exists('outputs/best_model.pth'):
        checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[20], gamma=0.1, verbose=True)

    train_loss_hist = Averager()
    train_loss_list, map_50_list, map_list = [], [], []
    metric = MeanAveragePrecision()

    MODEL_NAME = 'model'
    if VISUALIZE_TRANSFORMED_IMAGES:
        from train_utils import show_transformed_image

        show_transformed_image(train_loader)

    save_best_model = SaveBestModel()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")
        train_loss_hist.reset()
        start = time.time()
        train_loss = train(train_loader, model, optimizer, train_loss_hist)
        metric_summary = validate(valid_loader, model, metric)
        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch + 1} mAP@0.50:0.95: {metric_summary['map']}")
        print(f"Epoch #{epoch + 1} mAP@0.50: {metric_summary['map_50']}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])
        save_best_model(model, float(metric_summary['map']), epoch, 'outputs')
        save_model(epoch, model, optimizer)
        save_loss_plot(OUT_DIR, train_loss_list)
        save_mAP(OUT_DIR, map_50_list, map_list)
        scheduler.step()
