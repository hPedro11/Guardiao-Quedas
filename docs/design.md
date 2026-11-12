# Design: Guardião de Quedas

**Objetivo.** Entrega da GS 2026.2 (CitizenTech) em Cognitive Computing: aplicação de visão computacional que detecta queda de idosos pela webcam, com demo ao vivo, métricas comprovadas e relatório.

**Restrições.**
- Usar apenas técnicas do material do professor (Labs 01 a 14).
- Prazo de um dia: sem treino de modelo e sem dataset externo.
- Precisa rodar em notebook comum, em tempo real.

**Abordagens consideradas.**
1. *MediaPipe Pose + regra geométrica* (escolhida): sem treino, demo simples, regra explicável na arguição.
2. *Só subtração de fundo e proporção da silhueta*: usa menos conteúdo e falha quando a pessoa parada é absorvida pelo fundo do MOG2.
3. *YOLO pré-treinado + proporção da caixa*: mais pesado e a caixa não distingue deitar devagar de cair.

**Decisões.**
- Queda = descida rápida do quadril seguida de tronco na horizontal por 1 s.
- MOG2 entra como condição de movimento, não como silhueta, pelo motivo do item 2.
- Pré-processamento é opcional e a avaliação compara com e sem, para a análise crítica.
- Métricas por clipe (um clipe = um caso), calculadas sem bibliotecas extras.
- Relatório em PDF é gerado por script a partir das métricas reais, para não haver número inventado.

**Fora do escopo.** Envio de alerta por mensagem, várias pessoas, ajuste de limiares por aprendizado.
