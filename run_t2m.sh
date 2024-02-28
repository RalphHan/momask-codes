if [ $# -eq 1 ]; then
    docker run -itd \
      --name t2m \
      --gpus '"device=0"' \
      -v `pwd`:/momask-codes \
      ralphhan/priormdm \
      bash /momask-codes/$0 1
    exit 0
fi

cd /momask-codes
/opt/conda/envs/PriorMDM/bin/python train_t2m_transformer.py
