import streamlit as st
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, self.label_emb(labels)], dim=1)
        return self.model(x).view(-1, 1, 28, 28)

st.title("Digit Generator (0–9)")

digit = st.selectbox("Pick a digit", list(range(10)))
if st.button("Generate Images"):
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()

    z = torch.randn(5, 100)
    labels = torch.tensor([digit]*5)
    with torch.no_grad():
        images = model(z, labels).detach()

    for img in images:
        st.image(img.squeeze().numpy(), width=100, clamp=True)
