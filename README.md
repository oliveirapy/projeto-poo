# SpamScan

Projeto acadêmico de Programação Orientada a Objetos em Python para analisar mensagens de e-mail, SMS e comentários. Filtros pontuados e um classificador Naive Bayes devem indicar se uma mensagem parece legítima, suspeita ou spam, mostrando os motivos da análise.

## Estado atual

O repositório está em desenvolvimento. Até o momento, foi implementada a parte do Natan:

- `src/dominio/filtro.py`: classe abstrata `Filtro`, base `FiltroPorPadrao` e filtros de palavras-chave, links suspeitos, excesso de maiúsculas e remetente bloqueado.
- `src/servicos/filtro_bayes.py`: treinamento do Naive Bayes, leitura do CSV, estimativa de probabilidade, pontuação e explicação.
- `data/treino.csv`: exemplos em português rotulados como `spam` ou `legitima`.

As classes `Mensagem`, `Pontuacao` e as exceções do projeto ainda dependem da integração com as outras partes da equipe. Por isso, a aplicação completa e os testes integrados ainda não estão disponíveis.

## Estrutura prevista

```text
src/dominio/       mensagens, filtros, pontuações e vereditos
src/servicos/      análise, Naive Bayes, estatísticas e registro
src/persistencia/  persistência JSON
data/              dados de treino e exemplos
tests/             testes automatizados
```

## Parte do Natan

`Filtro.avaliar(mensagem)` é o método comum que chama `calcular()` e `motivo()` da classe concreta e monta uma `Pontuacao`. As regras são filtros explícitos escritos em Python. O Naive Bayes aprende frequências de palavras do CSV e estima a probabilidade de spam; não usa LLM nem serviço externo.

O classificador transforma a probabilidade em pontos considerando 50% como neutro: probabilidade de 50% gera zero pontos; 75% gera metade do peso; 100% gera o peso máximo. Essa escolha evita adicionar risco quando o modelo está indeciso.

## Próximas etapas da equipe

1. Integrar as classes de mensagem, pontuação e exceções.
2. Escrever testes para filtros e Naive Bayes e executar a suíte da equipe.
3. Completar analisador, persistência, estatísticas, menu e dados de demonstração.
4. Atualizar este README com o comando de execução e os resultados dos testes quando a aplicação estiver integrada.

O projeto usa apenas a biblioteca padrão do Python e tem como alvo Python 3.10 ou superior.
