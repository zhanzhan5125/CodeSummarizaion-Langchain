from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(model_output, attention_mask):
    """
    按照 README 示例，对 token embeddings 做 attention 加权平均。
    """
    token_embeddings = model_output[0]  # [batch_size, seq_len, hidden_dim]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(1),
                                                                            min=1e-9)  # :contentReference[oaicite:6]{index=6}


def side_score(code, summary, model_path, device=DEVICE):
    # 1. 加载 tokenizer 与模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # :contentReference[oaicite:7]{index=7}
    model = AutoModel.from_pretrained(model_path).to(device)  # :contentReference[oaicite:8]{index=8}

    sims = []
    # 2. 编码并计算 embeddings
    # print(code, summary)
    encoded = tokenizer([code, summary],
                        padding=True,
                        truncation=True,
                        return_tensors='pt').to(device)  # :contentReference[oaicite:9]{index=9}
    with torch.no_grad():
        output = model(**encoded)
    embs = mean_pooling(output, encoded['attention_mask'])
    embs = F.normalize(embs, p=2, dim=1)  # $\ell_2$ 归一化 :contentReference[oaicite:10]{index=10}

    # 3. 余弦相似度
    sim = util.pytorch_cos_sim(embs[0], embs[1]).item()  # :contentReference[oaicite:11]{index=11}
    sims.append(sim)
    # print(sim)

    return float(round(np.mean(sims) * 100, 4))


if __name__ == '__main__':
    print(side_score('''public Object pop() throws EmptyStackException {
  	    try {
  	      Object aObject = this.stackElements.get(stackElements.size() - 1);
  	      stackElements.remove(stackElements.size() - 1);
  	      return aObject;
  	    }
  	    catch (Exception e) {
  	      throw new EmptyStackException(e);
  	    }
		}''', "Returns a string consisting of the first n bytes from the queue without removing them.", './SIDE-Models/hard-negatives/141205'))
