# GCE Deployment & Setup Guide (`setup_gce.sh`)

This document explains exactly what the `setup_gce.sh` script does behind the scenes, followed by an end-to-end guide on how to launch the Souli pipeline on a Google Compute Engine (GCE) Virtual Machine.

---

## Part 1: How `setup_gce.sh` Works
The [scripts/setup_gce.sh](./scripts/setup_gce.sh) file is an automated Bash script designed to configure a brand new Ubuntu Linux VM to run the Souli AI pipeline. The entire pipeline requires Docker, NVIDIA GPUs (for fast transcription and local LLM inference), and the Ollama service to serve LLMs locally.

Here is a step-by-step breakdown of what the script does:

### 1. System Packages (`[1/6]`)
It updates the Ubuntu package list and installs fundamental tools required for downloading files, managing repositories, and handling audio/video processing: `curl`, `wget`, `git`, `unzip`, `ffmpeg` (for whisper audio processing), and `mpg123`.

### 2. NVIDIA GPU Drivers (`[2/6]`)
It checks if the VM has an NVIDIA GPU attached (`lspci | grep nvidia`). If it finds one, it installs the **NVIDIA CUDA Drivers** securely from NVIDIA's official repository. This is what allows the operating system itself to talk to the GPU hardware. If no GPU is found, it safely warns you and skips this step.

### 3. Docker & Container Toolkit (`[3/6]`)
- **Docker Core:** It downloads and installs the Docker engine using the official installation script. It also adds your user to the `docker` group so you don't have to type `sudo` before every docker command.
- **NVIDIA Container Toolkit:** Standard Docker isolating containers cannot see the host machine's GPU. The script installs the `nvidia-container-toolkit` and registers it with Docker. This allows our Docker containers (like Ollama and Faster-Whisper) to fully utilize the NVIDIA GPU for fast AI inference.

### 4. Docker Compose v2 (`[4/6]`)
Docker Compose lets us define and run multi-container setups (like our Souli pipeline + Qdrant + Ollama) through a single `docker-compose.yml` file. The script downloads the correct binary for Docker Compose v2 into the CLI plugins directory.

### 5. Clone the Repository (`[5/6]`)
It automatically pulls this entire codebase (`souli-voice-pipeline`) from GitHub and clones it into a standard Linux directory: `/opt/souli`. If the folder already exists, it just updates it (`git pull`). It also copies the `.env.example` file to create a `.env` file for you to configure later.

### 6. Pulling Ollama AI Models (`[6/6]`)
Finally, it boots up the local `ollama` Docker container in the background and instructs it to download the large AI models that the pipeline relies on:
- `llama3.1` (~4.7 GB) - Used for the conversation engine / Chatbot.
- `qwen2.5:1.5b` (~1 GB) - Used for tagging extracted YouTube chunks with energy nodes quickly.

---

## Part 2: End-to-End Deployment Guide

If you want to move the Souli Data Ingestion completely to GCP, here is your end-to-end workflow from creating the VM to talking to the AI.

### Step 1: Create the Virtual Machine
Create a VM instance in Google Cloud Console. 
- **Machine Type:** `n1-standard-4` (4 vCPUs, 15 GB RAM)
- **GPU:** 1 x NVIDIA T4
- **OS:** Ubuntu 22.04 LTS
- **Disk:** 100 GB SSD (AI models and YouTube videos take up significant space).
- **Firewall:** Allow HTTP/HTTPS traffic if you plan to expose the Streamlit UI, though you can cũng run everything via SSH (CLI).

### Step 2: Run the Setup Script
Once the VM is created and running, open the GCP Cloud Shell or your local terminal and run this single command to connect to the VM and execute the script remotely:

```bash
# Replace 'souli-vm' with the actual name of your VM in GCP
gcloud compute ssh souli-vm -- 'bash -s' < scripts/setup_gce.sh
```
*Go grab a coffee—this will take a few minutes to install drivers, docker, and download ~6GB of AI models.*

### Step 3: Configure your Environment
SSH into the machine manually to finish configuration:
```bash
gcloud compute ssh souli-vm
```
Edit the environment variables file using `nano`:
```bash
nano /opt/souli/.env
```
Add your required API keys (like the LiveKit variables if using voice, and any other configuration specific to your GCP environment).

### Step 4: Upload your Data
Your VM needs the raw Excel `gold` framework and your list of YouTube videos to process. You securely copy them from your local computer to the VM using `scp` (or `gcloud compute scp`):

```bash
# Run this ON YOUR LOCAL COMPUTER
gcloud compute scp "Souli_EnergyFramework_PW (1).xlsx" souli-vm:/opt/souli/data/
gcloud compute scp data/videos.csv souli-vm:/opt/souli/data/
```

### Step 5: Start the Database & APIs
On your VM, spin up the long-running services (Ollama and Qdrant) in detached mode (`-d`):
```bash
cd /opt/souli
docker compose -f docker-compose.gcp.yml up -d
```

### Step 6: Ingest the Data
Now run the full data pipeline inside the Docker environment. This processes all your uploaded YouTube videos, chunks them, and maps them to the uploaded Excel Framework.
*(Note: Ensure your pipeline.gcp.yaml paths align with `/app/data/` inside the container).*

```bash
docker compose -f docker-compose.gcp.yml exec souli \
  souli run all \
  --config /app/configs/pipeline.gcp.yaml \
  --videos-csv /app/data/videos.csv \
  --excel-path "/app/data/Souli_EnergyFramework_PW (1).xlsx" \
  --merge
```

### Step 7: Push to Vector DB
Once the pipeline produces the `teaching_ready.xlsx` outputs, ingest them into the Qdrant database running on the VM:
```bash
docker compose -f docker-compose.gcp.yml exec souli \
  souli ingest --config /app/configs/pipeline.gcp.yaml
```

### Step 8: Test the Chatbot
Everything is strictly local and running on your GPU. You can now test it via CLI:
```bash
docker compose -f docker-compose.gcp.yml exec souli \
  souli chat --config /app/configs/pipeline.gcp.yaml \
  --excel "/app/data/Souli_EnergyFramework_PW (1).xlsx"
```
Or start the voice application, or point your external Mobile Application to the GCP VM's IP address and exposed API port!