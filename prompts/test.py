import evaluate
from datasets.download import  download_config


config = download_config.DownloadConfig(proxies={
        "http":"http://127.0.0.1:7890",
        "https":"https://127.0.0.1:7890",

    })

if __name__ == '__main__':
    bleu = evaluate.load("../metric/bleu/bleu.py")
    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = [
        ["hello there general kenobi", "hello there !"],
        ["foo bar foobar"]
        ]
    results = bleu.compute(predictions=predictions, references=references)
    print(results)