import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.optim as optim
from torch_geometric.data import Data, DataLoader as GraphDataLoader
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import BertTokenizer, BertModel
from models import ClipGraph, GATMod
from losses import square_contrastive_loss
import ast

# 1. 数据预处理
text_data = pd.read_csv(r'F:\learningfiles\myproject\congrat-master\prohect_test2\data\大学课程数据集（概念+描述）.csv')
concept_names = text_data['概念名称'].tolist()
concept_descriptions = text_data['概念描述'].tolist()

nodes = pd.read_csv(r'F:\learningfiles\myproject\congrat-master\prohect_test2\data\node.csv')
edges = pd.read_csv(r'F:\learningfiles\myproject\congrat-master\prohect_test2\data\edge.csv')
nodes['vector'] = nodes['vector'].apply(ast.literal_eval)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(concept_descriptions, return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 创建Tensor数据集和DataLoader
dataset = TensorDataset(input_ids, attention_mask)
batch_size = 22
text_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

concept_to_id = {name: i for i, name in enumerate(nodes['概念名称'].tolist())}
edge_index = []
for src, tgt in zip(edges['Source'], edges['Target']):
    edge_index.append([concept_to_id[src], concept_to_id[tgt]])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
graph_x = torch.tensor(nodes['vector'].tolist(), dtype=torch.float)
graph_data = Data(x=graph_x, edge_index=edge_index)


# 将graph_data转换为一个列表，其中每个元素都是一个Data对象
graph_data_list = [Data(x=graph_x[i:i+batch_size], edge_index=edge_index) for i in range(0, len(graph_x), batch_size)]

# 创建Graph DataLoader
graph_dataloader = GraphDataLoader(graph_data_list, batch_size=1, shuffle=True)  # 注意：batch_size设置为1，因为我们已经手动分批了


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClipGraph(
    lm=BertModel.from_pretrained('bert-base-uncased'),
    gnn=GATMod(in_channels=768, out_channels=768)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for (batch_input_ids, batch_attention_mask), batch_graph_data in zip(text_dataloader, graph_dataloader):
        optimizer.zero_grad()

        # 由于batch_size设置为1，我们需要从列表中提取Data对象
        batch_graph_data = batch_graph_data[0]

        # 创建node_index_tensor
        batch_node_index_tensor = torch.arange(batch_graph_data.x.size(0)).to(device)

        logits = model(batch_input_ids.to(device), batch_attention_mask.to(device), batch_graph_data.x.to(device),
                       batch_graph_data.edge_index.to(device), batch_node_index_tensor)

        loss = square_contrastive_loss(logits)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

print("Training complete!")

model.eval()
with torch.no_grad():
    text_embeddings = model.embed_text(input_ids.to(device), attention_mask.to(device))
    graph_embeddings = model.embed_nodes(graph_data.x.to(device), graph_data.edge_index.to(device))
    joint_embeddings = text_embeddings + graph_embeddings

embeddings_df = pd.DataFrame({
    'concept_name': concept_names,
    'joint_embedding': joint_embeddings.cpu().numpy().tolist()
})

embeddings_df.to_csv('joint_embeddings.csv', index=False)
print("Joint embeddings saved to joint_embeddings.csv!")
