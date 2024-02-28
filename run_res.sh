if [ $# -eq 0 ]; then
    docker run -itd \
      --name res \
      --gpus '"device=1"' \
      -v `pwd`:/momask-codes \
      ralphhan/priormdm \
      bash /momask-codes/$0 1
    exit 0
fi

cd /momask-codes
/opt/conda/envs/PriorMDM/bin/python train_res_transformer.py
