#!/usr/bin/env python
"""
@author: metalcorebear
"""

import newsplex
import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--event_date', help='Enter the event date as "YYYY-MM-dd".', required=True)
    parser.add_argument('-o', '--output', help='Enter the output path.', required=False)
    parser.add_argument('-k', '--key_location', help='Enter the API Key location.', required=False)
    args = vars(parser.parse_args())
    output_path = str(args['output'])
    key_location = str(args['key_location'])
    event_date = str(args['event_date'])
    return output_path, key_location, event_date

if __name__ == '__main__':
    output_path, key_location, event_date = get_params()
    aggregator = newsplex.newsaggregator(key_location=key_location, filepath=output_path)
    aggregator.run_query(event_date)
    aggregator.generate_json()

    news_results = newsplex.newsplex(filepath=output_path)
    news_results.fit()
    news_results.plot_embeddings()
    news_results.calculate_persistences()

    print('Topology is fun!')
    