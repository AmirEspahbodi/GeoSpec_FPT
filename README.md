python train.py \
    batch_size=16 \
    epochs=100 \
    dataset.input_size=512 \
    dataset.learning_rate=0.0005 \
    network.num_prompts=16 \
    network.side_input_size=128

Get-Service -Name ssh-agent | Set-Service -StartupType Manual
Start-Service ssh-agent
ssh-add C:\Users\amir\.ssh\id_ed25519.github
https://github.com/YubiaoYue/MedMamba/blob/main/MedMamba.py

python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install mamba-ssm --no-build-isolation
