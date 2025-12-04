import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def export_logs(log_dir="runs"):
    print(f"Searching for events in: {log_dir}")
    
    files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "tfevents" in f]
    if not files:
        print("No TensorBoard event files found.")
        return
        
    latest_file = max(files, key=os.path.getctime)
    print(f"Processing: {latest_file}")
    
    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    print(f"Found tags: {tags}")

    # Extract Step-Level Metrics (Batch Loss, LR) 
    step_data = None 
    step_tags = ["Train/BatchLoss", "Train/LearningRate"]
    
    for tag in step_tags:
        if tag in tags:
            events = ea.Scalars(tag)
            temp_df = pd.DataFrame([(x.step, x.value) for x in events], columns=["step", tag])
            
            if step_data is not None:
                step_data = pd.merge(step_data, temp_df, on="step", how="outer")
            else:
                step_data = temp_df
    
    if step_data is not None and not step_data.empty:
        step_data = step_data.sort_values("step")
        step_data.to_csv("training_log_steps.csv", index=False)
        print(f"Saved step-level metrics to: training_log_steps.csv ({len(step_data)} rows)")

    # Extract Epoch-Level Metrics (Train/Val Loss)
    epoch_data = None
    epoch_tags = ["Epoch/TrainLoss", "Epoch/ValLoss"]
    
    for tag in epoch_tags:
        if tag in tags:
            events = ea.Scalars(tag)
            temp_df = pd.DataFrame([(x.step, x.value) for x in events], columns=["epoch", tag])
            
            if epoch_data is not None:
                epoch_data = pd.merge(epoch_data, temp_df, on="epoch", how="outer")
            else:
                epoch_data = temp_df

    if epoch_data is not None and not epoch_data.empty:
        epoch_data = epoch_data.sort_values("epoch")
        epoch_data.to_csv("training_log_epochs.csv", index=False)
        print(f"Saved epoch-level metrics to: training_log_epochs.csv ({len(epoch_data)} rows)")

if __name__ == "__main__":
    export_logs()
