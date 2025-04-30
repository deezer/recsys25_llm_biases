import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import trange

from evaluator import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer, util, models


def get_sentence_transformer_embeddings(model_name, corpus, corpus_chunk_size=10):
	model = SentenceTransformer(model_name)

	text_embs = []
	if len(corpus) < corpus_chunk_size:
		text_embs = model.encode(corpus, show_progress_bar=True, convert_to_tensor=True).detach().cpu()
	else:
		for corpus_start_idx in trange(0, len(corpus), corpus_chunk_size, desc="Corpus Chunks", disable=False):
        	
			corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
			sub_corpus_embeddings =  model.encode(corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, convert_to_tensor=True)
			text_embs.extend(sub_corpus_embeddings.detach().cpu())
		text_embs = torch.stack(text_embs)
	return text_embs



if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_path', dest='output_path', type=str, required=True, help='Path where to save the results')
	parser.add_argument('--test_data_path', dest='test_data_path', type=str, required=True, help='Path where to find the test files')
	parser.add_argument('--our_model_path', dest='our_model_path', type=str, required=True, help='Path where to find our model')
	
	args = parser.parse_args()

	with open(f'{args.test_data_path}queries.json', 'r') as _:
		queries = json.load(_)

	queries_ids = list(queries.keys())
	queries_content = [queries[qid] for qid in queries_ids]
	embs = get_sentence_transformer_embeddings(args.our_model_path, queries_content)
	query_embeddings = dict(zip(queries_ids, embs))

	with open(f'{args.test_data_path}positives_per_user.json', 'r') as _:
		positives_per_user = json.load(_)

	results = {}
	for seed in [42, 0, 1, 2, 3]:	
		with open(f'{args.test_data_path}corpus_{seed}.json', 'r') as _:
			tracks = json.load(_)
		corpus_ids = list(tracks.keys())
		corpus_content = [tracks[track_id] for track_id in tracks]
		embs = get_sentence_transformer_embeddings(args.our_model_path, corpus_content)
		corpus_embeddings = dict(zip(corpus_ids, embs))

		results = {}
		for key in queries:
			user_id, m, tw = key.split('/')

			if user_id not in positives_per_user:
				continue

			user_queries = {}
			user_queries[user_id] = queries[key]
			user_query_embeddings = [query_embeddings[key]]
			user_query_embeddings = torch.stack(user_query_embeddings)

			relevant_tracks = {} # mapping between query and tracks
			relevant_tracks[user_id] = positives_per_user[user_id]

			user_corpus = {}
			for track_id in positives_per_user[user_id]:
				user_corpus[track_id] = corpus[track_id]
					
			user_corpus_ids = list(user_corpus.keys())
			user_corpus_embeddings = [corpus_embeddings[cid] for cid in user_corpus_ids]
			user_corpus_embeddings = torch.stack(user_corpus_embeddings) 

			ire = InformationRetrievalEvaluator(user_queries, user_corpus, relevant_tracks)
			results[key] = ire.compute_metrices(user_corpus_embeddings, user_query_embeddings)
			print(rkey, results[key])

		os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
		with open(f"{args.output_path}results_seed{seed}.csv", 'w') as _:
			_.write("user_id, time_window, model, recall@10, ndcg@10\n")
			for rkey in results:
				user_id, m, tw = rkey.split('/')
				recall10 = results[rkey]['cos_sim']['recall@k'][10]
				ndcg10 = results[rkey]['cos_sim']['ndcg@k'][10]
				_.write(','.join([user_id, tw, m, "{:.5f}".format(recall10), "{:.5f}".format(ndcg10)]) + "\n")
