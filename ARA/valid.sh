CUDA_VISIBLE_DEVICES="1" \
python -m colbert.test --amp --doc_maxlen 180 --mask-punctuation \
--topk ./data/MSMARCO/top1000.dev  \
--checkpoint ./experiments/MSMARCO-psg/train.py/msmarco.psg.ll/checkpoints/colbert-60000.dnn \
--root ./experiments/ --experiment MSMARCO-psg --qrels ./data/MSMARCO/qrels.dev.tsv --similarity l2 --query_maxlen 96
