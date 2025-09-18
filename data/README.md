## Dataset description

The training data includes three mandatory files, but more will be generated automatically in the training pipeline:

1. **`corpus.jsonl`**  
   Contains the music catalog in JSONL format. Each entry represents a track with metadata description.  
   Example:  
   ```json
   {
     "_id": "98981",
     "title": "",
     "text": "title: Search and Destroy\n\tartist: The Stooges\n\torigin country: United States of America\n\talbum release date: 2011.0\n\tmain genre: ('Rock music',)\n\tsecondary genre: ('Hard rock', 'Punk rock', 'Rock music', 'US alternative rock')"
   }

2. **`qgen-queries.jsonl`**  
	Contains user-generated profiles that act as queries in the recommendation task. These profiles describe users' musical preferences.
	Example:

	```json
 	{
	  "_id": "35656_0",
	  "metadata": {},
	  "text": "The user's musical taste is characterized by a strong preference for rock music with diverse subgenres such as punk rock, blues rock, hard rock, metal, and alternative rock. They also enjoy elements of pop music and have a liking for remastered albums like \"Dazed And Confused (2007 Remastered LP Version)\". Their tastes span across multiple eras and include both classic and modern artists or bands, indicating a eclectic and versatile musical palette."
	}

3. **`qgen-qrels/train.tsv`**  
	Provides the ground-truth relevance mappings between user profiles (queries) and tracks (corpus). It is a tab-separated file with records of the form:

	```query-id    corpus-id    score```


	Example:

	```35656_0    98981    1```

Similarly, the testing dataset also relies on three mandatory files:

1. **`corpus_seed.json`**  
Defines the catalog of tracks available in the test set. Its structure is similar to `corpus.jsonl`.

2. **`queries.json`**  
Contains user query profiles for the test set, similar in structure to `qgen-queries.jsonl`.

3. **`positives_per_user.json`**  
A JSON file mapping each user (query) to a list of relevant tracks.

Example:

```json
{
  "35656_0": [
    1001,
    14576,
    562732,
    24081,
  ]
}
