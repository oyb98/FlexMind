CUDA_VISIBLE_DEVICES="1" \
python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 \
--triples ./data/MSMARCO/triples.DQ.sample.train.small.tsv \
--root ./experiments/ --experiment MSMARCO-psg --similarity l2 --query_maxlen 96 --run msmarco.psg.dq