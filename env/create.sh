conda create -y -n spml python=3.9 && \
conda install -y -n spml pytorch=1.9.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia && \
conda run -n spml --no-capture-output pip install -r requirements.txt
