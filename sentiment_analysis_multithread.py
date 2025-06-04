import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import time


def init_process():
    global vader_analyzer, afinn_analyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    afinn_analyzer = Afinn()



def analyze_chunk(args):
    chunk, _ = args
    #VADER
    texts = chunk['soft_cleaned_text'].values.astype('U')
    vader_scores = np.array([vader_analyzer.polarity_scores(text)['compound'] for text in texts])

    #AFINN
    afinn_scores = np.fromiter(
        (afinn_analyzer.score(text) for text in texts),
        dtype=np.int16
    )

    chunk['vader_sentiment'] = vader_scores
    chunk['afinn_sentiment'] = afinn_scores
    return chunk


def main():
    #i9-13900H
    PHYSICAL_CORES = 6
    EFFICIENCY_CORES = 8
    TOTAL_THREADS = 20
    MEMORY_GB = 32


    input_file = 'double_cleaned_2022.csv'
    output_file = 'analysed_2022.csv'
    optimal_chunksize = 50000
    process_workers = PHYSICAL_CORES + 2


    reader = pd.read_csv(
        input_file,
        chunksize=optimal_chunksize,
        dtype={'soft_cleaned_text': 'string'},
        engine='c',
        low_memory=True
    )


    with ProcessPoolExecutor(
            max_workers=process_workers,
            initializer=init_process
    ) as executor:

        total_chunks = (15_000_000 // optimal_chunksize) + 1
        tasks = ((chunk, idx) for idx, chunk in enumerate(reader))


        results = list(tqdm(
            executor.map(analyze_chunk, tasks, chunksize=2),
            total=total_chunks,
            desc="Processing",
            unit="chunk"
        ))


    final_df = pd.concat(results)


    final_df.to_csv(output_file, index=False, compression='gzip')
    print("处理完成")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"总耗时: {time.time() - start_time:.2f}秒")