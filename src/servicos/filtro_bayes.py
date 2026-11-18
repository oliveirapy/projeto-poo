"""Etapa de treinamento do filtro Naive Bayes do SpamScan."""

from collections import Counter
import csv
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING

from src.dominio.filtro import Filtro
from src.excecoes import ErroFiltro, ErroValidacao

if TYPE_CHECKING:
    from src.dominio.mensagem import Mensagem


class FiltroNaiveBayes(Filtro):
    """Armazena frequências de palavras aprendidas a partir de exemplos rotulados."""

    def __init__(self, peso: float = 4.0) -> None:
        """Cria o filtro vazio; é necessário chamar treinar() antes de usá-lo."""
        super().__init__("Naive Bayes", peso)
        self._contagens_palavras: dict[str, Counter[str]] = {
            "spam": Counter(),
            "legitima": Counter(),
        }
        self._quantidade_mensagens: Counter[str] = Counter()
        self._vocabulario: set[str] = set()
        self._treinado = False

    @staticmethod
    def _tokenizar(texto: str) -> list[str]:
        """Divide texto em palavras minúsculas, incluindo letras acentuadas."""
        if not isinstance(texto, str):
            raise ErroValidacao("O texto para tokenização precisa ser uma string.")
        return re.findall(r"[^\W\d_]+", texto.casefold(), flags=re.UNICODE)

    def treinar(self, exemplos: list[tuple[str, str]]) -> None:
        """Aprende contagens a partir de pares (texto, classe).

        As classes aceitas são 'spam' e 'legitima'. O método monta as novas
        contagens localmente e só atualiza o objeto depois que todo o treino
        passou pela validação.
        """
        if not exemplos:
            raise ErroValidacao("O treinamento precisa ter pelo menos um exemplo.")

        contagens: dict[str, Counter[str]] = {
            "spam": Counter(),
            "legitima": Counter(),
        }
        quantidade: Counter[str] = Counter()
        vocabulario: set[str] = set()

        for numero, exemplo in enumerate(exemplos, start=1):
            if not isinstance(exemplo, (tuple, list)) or len(exemplo) != 2:
                raise ErroValidacao(
                    f"O exemplo {numero} precisa ter o formato (texto, classe)."
                )

            texto, classe = exemplo
            if not isinstance(texto, str) or not isinstance(classe, str):
                raise ErroValidacao(f"Texto e classe do exemplo {numero} precisam ser strings.")

            classe_normalizada = classe.strip().casefold()
            if classe_normalizada not in contagens:
                raise ErroValidacao(
                    f"Classe inválida no exemplo {numero}: {classe!r}. "
                    "Use 'spam' ou 'legitima'."
                )

            palavras = self._tokenizar(texto)
            if not palavras:
                raise ErroValidacao(f"O texto do exemplo {numero} não contém palavras.")

            contagens[classe_normalizada].update(palavras)
            quantidade[classe_normalizada] += 1
            vocabulario.update(palavras)

        if not all(quantidade[classe] > 0 for classe in contagens):
            raise ErroValidacao(
                "O treinamento precisa conter exemplos de spam e de mensagens legitimas."
            )

        self._contagens_palavras = contagens
        self._quantidade_mensagens = quantidade
        self._vocabulario = vocabulario
        self._treinado = True

    def treinar_csv(self, caminho: str | Path) -> None:
        """Lê um CSV com colunas 'texto' e 'classe' e treina o classificador."""
        caminho = Path(caminho)
        try:
            with caminho.open("r", encoding="utf-8", newline="") as arquivo:
                leitor = csv.DictReader(arquivo)
                if leitor.fieldnames is None or not {"texto", "classe"}.issubset(leitor.fieldnames):
                    raise ErroValidacao(
                        "O CSV precisa ter as colunas 'texto' e 'classe'."
                    )
                exemplos = [(linha["texto"], linha["classe"]) for linha in leitor]
        except (OSError, UnicodeError, csv.Error) as erro:
            raise ErroValidacao(f"Não foi possível ler o arquivo de treinamento {caminho}.") from erro

        self.treinar(exemplos)

    def _log_probabilidades(self, texto: str) -> dict[str, float]:
        """Calcula o log da evidência do texto para cada classe treinada."""
        if not self._treinado:
            raise ErroFiltro("O Naive Bayes precisa ser treinado antes da análise.")

        palavras = self._tokenizar(texto)
        frequencias_texto = Counter(palavras)
        total_mensagens = sum(self._quantidade_mensagens.values())
        tamanho_vocabulario = len(self._vocabulario)
        resultados: dict[str, float] = {}

        for classe in ("spam", "legitima"):
            # A priori: chance da classe antes de observar as palavras.
            probabilidade_classe = self._quantidade_mensagens[classe] / total_mensagens
            log_score = math.log(probabilidade_classe)

            # O denominador e o +1 implementam a suavização de Laplace.
            total_palavras_classe = sum(self._contagens_palavras[classe].values())
            denominador = total_palavras_classe + tamanho_vocabulario

            for palavra, quantidade in frequencias_texto.items():
                ocorrencias = self._contagens_palavras[classe][palavra]
                probabilidade_palavra = (ocorrencias + 1) / denominador
                log_score += quantidade * math.log(probabilidade_palavra)

            resultados[classe] = log_score

        return resultados

    def probabilidade_spam(self, texto: str) -> float:
        """Converte as evidências das classes em probabilidade de spam (0 a 1)."""
        scores = self._log_probabilidades(texto)
        diferenca = scores["spam"] - scores["legitima"]

        # Forma estável da função logística para evitar overflow em exp().
        if diferenca >= 0:
            return 1 / (1 + math.exp(-diferenca))
        exp_diferenca = math.exp(diferenca)
        return exp_diferenca / (1 + exp_diferenca)

    def calcular(self, mensagem: "Mensagem") -> float:
        """Transforma probabilidade em pontos, tratando 50% como neutro."""
        probabilidade = self.probabilidade_spam(mensagem.texto_analisavel())
        # p=0.5 resulta em zero; p=1 resulta no peso total do filtro.
        return self.peso * max(0.0, 2 * probabilidade - 1)

    def _palavras_mais_influentes(self, texto: str, limite: int = 3) -> list[tuple[str, str]]:
        """Seleciona palavras que mais favorecem spam ou mensagem legítima."""
        palavras = Counter(self._tokenizar(texto))
        tamanho_vocabulario = len(self._vocabulario)
        total_spam = sum(self._contagens_palavras["spam"].values()) + tamanho_vocabulario
        total_legitima = sum(self._contagens_palavras["legitima"].values()) + tamanho_vocabulario

        influencias = []
        for palavra, quantidade in palavras.items():
            prob_spam = (self._contagens_palavras["spam"][palavra] + 1) / total_spam
            prob_legitima = (self._contagens_palavras["legitima"][palavra] + 1) / total_legitima
            influencia = quantidade * math.log(prob_spam / prob_legitima)
            classe_favorecida = "spam" if influencia > 0 else "legitima"
            influencias.append((palavra, classe_favorecida, abs(influencia)))

        influencias.sort(key=lambda item: item[2], reverse=True)
        return [(palavra, classe) for palavra, classe, _ in influencias[:limite]]

    def motivo(self, mensagem: "Mensagem") -> str:
        """Explica a probabilidade estimada e destaca palavras influentes."""
        texto = mensagem.texto_analisavel()
        probabilidade = self.probabilidade_spam(texto)
        evidencias = self._palavras_mais_influentes(texto)

        if evidencias:
            lista = ", ".join(
                f"{palavra!r} (favorece {classe})" for palavra, classe in evidencias
            )
        else:
            lista = "nenhuma palavra para comparar"

        return (
            f"Probabilidade estimada de spam: {probabilidade:.0%}. "
            f"Palavras mais influentes: {lista}."
        )
