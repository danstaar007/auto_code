### HOW TO RUN THIS APPLICATION ###

1️⃣ **Training the Model (On GPU Machines)**
   - Start the Dask cluster:
     ```
     python dask_setup.py
     ```
   - Load and preprocess dataset:
     ```
     python dataset_loader.py
     ```
   - Train the model:
     ```
     python train_model.py --ddp   # For multi-GPU DDP training
     python train_model.py --cpu   # For CPU offloading (Threadripper + 7950X3D)
     python train_model.py --fsdp  # For Fully Sharded Data Parallel (FSDP)
     ```
   - Monitor training with TensorBoard:
     ```
     tensorboard --logdir=tensorboard_logs --host=0.0.0.0 --port=6006
     ```

2️⃣ **Running Inference on a CPU-Only Machine**
   - Copy the trained model (`checkpoints/`) to the new machine.
   - Run the inference script:
     ```
     python cpu_inference.py
     ```
   - Enter a code prompt when prompted.

3️⃣ **Using Hotkey-Based Code Completion (On Any Machine)**
   - Run the script:
     ```
     python hotkey_completion.py
     ```
   - Press `Ctrl+Shift+C` to auto-generate code.

4️⃣ **Running the Code Completion API (For External Use)**
   - Start the API:
     ```
     python api_server.py
     ```
   - Test API with:
     ```
     curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"prompt": "def fibonacci(n):", "max_length": 100}'
     ```
