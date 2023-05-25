# Ordrec
Ordrec tries to solve data distribution heterogenity problem.

# Todo
- Make trainer
- Train & valid split
- Test function

# Reference
1. [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch)
2. [OrderedGNN](https://github.com/LUMIA-Group/OrderedGNN)

# Note

### 05/25
- Adjacency matrix normalization 할 필요 없음 -> message passing aggregation method가 mean pooling
- 현재 학습 구조는 full graph를 학습한 후 masking하는 형태 -> minibatch 사용함 -> user ID로 data loader 만들면 됨
- GONN에서 마지막 linear layer 필요?

