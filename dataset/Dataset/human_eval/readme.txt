This directory contains the human evaluation dataset used in our experiment.

1. java.txt, python.txt, c.txt:
	Each contains 50 samples. 
	Each sample contains a piece of code and the corresponding five code summaries Summary 1~Summary 5.

2. human_eval_record_{language}.csv: 
	These files record the index of each sample in the original java/python/c dataset, 
	and the corresponding LLM of Summary 1~Summary 5,
	to facilitate subsequent collation of human evaluation results.