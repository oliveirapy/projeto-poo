"""Abstrações compartilhadas pelos filtros do SpamScan."""

from abc import ABC, abstractmethod
import ipaddress
import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from src.excecoes import ErroFiltro, ErroValidacao
from src.dominio.pontuacao import Pontuacao

if TYPE_CHECKING:
    from src.dominio.mensagem import Mensagem


class Filtro(ABC):
    """Contrato e método-modelo para avaliar uma mensagem."""

    def __init__(self, nome: str, peso: float) -> None:
        """Guarda o nome e o peso após validar os valores recebidos."""
        if not isinstance(nome, str) or not nome.strip():
            raise ErroValidacao("O nome do filtro não pode ser vazio.")
        if isinstance(peso, bool) or not isinstance(peso, (int, float)):
            raise ErroValidacao("O peso do filtro precisa ser numérico.")
        if peso <= 0:
            raise ErroValidacao("O peso do filtro precisa ser maior que zero.")

        self._nome = nome.strip()
        self._peso = float(peso)

    @property
    def nome(self) -> str:
        """Nome usado para identificar o filtro no resultado da análise."""
        return self._nome

    @property
    def peso(self) -> float:
        """Pontuação máxima que este filtro pode atribuir."""
        return self._peso

    def avaliar(self, mensagem: "Mensagem") -> Pontuacao:
        """Executa o filtro e empacota valor e explicação em Pontuacao.

        Este método-modelo é igual para todos os filtros concretos. As
        subclasses fornecem calcular() e motivo().
        """
        try:
            valor = self.calcular(mensagem)
            if isinstance(valor, bool) or not isinstance(valor, (int, float)):
                raise ErroValidacao("O cálculo do filtro precisa retornar um número.")
            if not 0 <= valor <= self.peso:
                raise ErroValidacao(
                    f"A pontuação de {self.nome!r} precisa estar entre 0 e {self.peso}."
                )

            explicacao = self.motivo(mensagem)
            if not isinstance(explicacao, str) or not explicacao.strip():
                raise ErroValidacao("O motivo do filtro precisa ser um texto não vazio.")

            return Pontuacao(self.nome, float(valor), explicacao.strip())
        except ErroFiltro:
            raise
        except Exception as erro:
            raise ErroFiltro(f"Falha ao avaliar o filtro {self.nome!r}.") from erro

    @abstractmethod
    def calcular(self, mensagem: "Mensagem") -> float:
        """Calcula o risco indicado pelo filtro, dentro do intervalo [0, peso]."""

    @abstractmethod
    def motivo(self, mensagem: "Mensagem") -> str:
        """Descreve a evidência que levou à pontuação."""


class FiltroPorPadrao(Filtro):
    """Classe intermediária para filtros baseados em expressões regulares."""

    def __init__(self, nome: str, peso: float, padroes: list[str]) -> None:
        """Compila os padrões uma vez, quando o filtro é criado."""
        super().__init__(nome, peso)
        try:
            self._padroes = [re.compile(padrao, re.IGNORECASE) for padrao in padroes]
        except re.error as erro:
            raise ErroValidacao(f"Expressão regular inválida: {erro}") from erro

    def _ocorrencias(self, texto: str) -> list[str]:
        """Retorna todos os trechos do texto que correspondem aos padrões."""
        return [
            correspondencia.group(0)
            for padrao in self._padroes
            for correspondencia in padrao.finditer(texto)
        ]


class FiltroPalavrasChave(FiltroPorPadrao):
    """Pontua mensagens que contêm palavras ou expressões de alerta."""

    _PALAVRAS_PADRAO = [
        "ganhe",
        "prêmio",
        "premio",
        "grátis",
        "gratis",
        "clique aqui",
        "renda extra",
        "urgente",
        "empréstimo",
        "emprestimo",
        "lucro garantido",
        "resgate",
    ]

    def __init__(self, peso: float = 3.0, palavras: list[str] | None = None) -> None:
        """Cria o filtro usando termos padrão ou uma lista personalizada."""
        termos = self._PALAVRAS_PADRAO if palavras is None else palavras
        if not termos:
            raise ErroValidacao("Informe pelo menos uma palavra ou expressão para o filtro.")
        if any(not isinstance(termo, str) or not termo.strip() for termo in termos):
            raise ErroValidacao("Cada palavra-chave precisa ser um texto não vazio.")

        self._termos = [termo.strip() for termo in termos]
        padroes = [rf"\b{re.escape(termo)}\b" for termo in self._termos]
        super().__init__("Palavras-chave", peso, padroes)

    def calcular(self, mensagem: "Mensagem") -> float:
        """Dá metade do peso por ocorrência, até atingir o peso máximo."""
        ocorrencias = self._ocorrencias(mensagem.texto_analisavel())
        pontos_por_ocorrencia = self.peso / 2
        return min(self.peso, len(ocorrencias) * pontos_por_ocorrencia)

    def motivo(self, mensagem: "Mensagem") -> str:
        """Lista os termos encontrados ou informa que nenhum foi localizado."""
        ocorrencias = self._ocorrencias(mensagem.texto_analisavel())
        if not ocorrencias:
            return "Nenhuma palavra-chave de alerta foi encontrada."
        return "Termos encontrados: " + ", ".join(ocorrencias)


class FiltroLinksSuspeitos(FiltroPorPadrao):
    """Atribui risco a links encurtados ou com características suspeitas."""

    _ENCURTADORES = {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "cutt.ly",
        "rb.gy",
        "is.gd",
    }
    _PONTUACAO_SINAL = {
        "encurtador": 1.0,
        "ip": 1.0,
        "http": 0.25,
        "punycode": 0.75,
    }

    def __init__(self, peso: float = 3.0) -> None:
        """Configura os padrões usados para encontrar URLs na mensagem."""
        padroes = [r"https?://[^\s<>\"']+", r"\bwww\.[^\s<>\"']+"]
        super().__init__("Links suspeitos", peso, padroes)

    @staticmethod
    def _limpar_link(link: str) -> str:
        """Remove pontuação final que normalmente pertence à frase."""
        return link.rstrip(".,;:!?)]}")

    def _sinais_do_link(self, link: str) -> list[str]:
        """Retorna os sinais de risco identificados em um link."""
        link = self._limpar_link(link)
        url = link if "://" in link else f"//{link}"
        host = (urlsplit(url).hostname or "").casefold().rstrip(".")
        sinais = []

        if any(host == dominio or host.endswith(f".{dominio}")
               for dominio in self._ENCURTADORES):
            sinais.append("encurtador")

        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            sinais.append("ip")

        if host.startswith("xn--") or ".xn--" in host:
            sinais.append("punycode")

        if link.casefold().startswith("http://"):
            sinais.append("http")

        return sinais

    def _links_com_sinais(self, mensagem: "Mensagem") -> list[tuple[str, list[str]]]:
        """Associa cada URL encontrada aos sinais de risco detectados."""
        links = self._ocorrencias(mensagem.texto_analisavel())
        return [(self._limpar_link(link), self._sinais_do_link(link)) for link in links]

    def calcular(self, mensagem: "Mensagem") -> float:
        """Soma o risco dos sinais e limita o total ao peso do filtro."""
        total = 0.0
        for _, sinais in self._links_com_sinais(mensagem):
            total += sum(self._PONTUACAO_SINAL[sinal] for sinal in sinais)
        return min(self.peso, total)

    def motivo(self, mensagem: "Mensagem") -> str:
        """Explica os links encontrados e os sinais associados a cada um."""
        encontrados = self._links_com_sinais(mensagem)
        if not encontrados:
            return "Nenhum link foi encontrado."

        suspeitos = [(link, sinais) for link, sinais in encontrados if sinais]
        if not suspeitos:
            return f"Foram encontrados {len(encontrados)} link(s), sem sinais de risco configurados."

        descricoes = {
            "encurtador": "encurtador conhecido",
            "ip": "endereço IP usado como host",
            "http": "conexão sem HTTPS",
            "punycode": "domínio com codificação punycode",
        }
        partes = [
            f"{link} ({', '.join(descricoes[sinal] for sinal in sinais)})"
            for link, sinais in suspeitos
        ]
        return "Sinais suspeitos: " + "; ".join(partes)


class FiltroExcessoMaiusculas(Filtro):
    """Pontua mensagens em que uma parcela alta das letras está em maiúsculas."""

    def __init__(self, peso: float = 2.0, limite: float = 0.65) -> None:
        """Cria o filtro e define a proporção a partir da qual há alerta."""
        super().__init__("Excesso de maiúsculas", peso)
        if isinstance(limite, bool) or not isinstance(limite, (int, float)):
            raise ErroValidacao("O limite precisa ser numérico.")
        if not 0 < limite < 1:
            raise ErroValidacao("O limite precisa estar entre 0 e 1, sem incluir os extremos.")
        self._limite = float(limite)

    def _proporcao_maiusculas(self, mensagem: "Mensagem") -> float:
        """Calcula a fração de letras maiúsculas, ignorando espaços e sinais."""
        letras = [caractere for caractere in mensagem.texto_analisavel()
                  if caractere.isalpha()]
        if not letras:
            return 0.0
        maiusculas = sum(1 for caractere in letras if caractere.isupper())
        return maiusculas / len(letras)

    def calcular(self, mensagem: "Mensagem") -> float:
        """Atribui zero até o limite e escala gradualmente até o peso máximo."""
        proporcao = self._proporcao_maiusculas(mensagem)
        if proporcao <= self._limite:
            return 0.0
        escala = (proporcao - self._limite) / (1 - self._limite)
        return self.peso * escala

    def motivo(self, mensagem: "Mensagem") -> str:
        """Mostra a proporção de letras maiúsculas e o limite configurado."""
        proporcao = self._proporcao_maiusculas(mensagem)
        percentual = proporcao * 100
        limite_percentual = self._limite * 100
        if proporcao <= self._limite:
            return (f"{percentual:.0f}% das letras são maiúsculas; "
                    f"o alerta começa acima de {limite_percentual:.0f}%.")
        return (f"{percentual:.0f}% das letras são maiúsculas, acima do limite "
                f"de {limite_percentual:.0f}%.")


class FiltroRemetenteBloqueado(Filtro):
    """Pontua mensagens cujo remetente está numa lista de bloqueio."""

    def __init__(self, remetentes_bloqueados: set[str] | None = None,
                 peso: float = 4.0) -> None:
        """Guarda remetentes normalizados para comparação sem diferença de caixa."""
        super().__init__("Remetente bloqueado", peso)
        remetentes = remetentes_bloqueados or set()
        if any(not isinstance(remetente, str) or not remetente.strip()
               for remetente in remetentes):
            raise ErroValidacao("A lista de remetentes contém um valor vazio ou inválido.")
        self._remetentes_bloqueados = {
            remetente.strip().casefold() for remetente in remetentes
        }

    def calcular(self, mensagem: "Mensagem") -> float:
        """Retorna o peso inteiro se o remetente estiver bloqueado; caso contrário, zero."""
        remetente = mensagem.remetente.strip().casefold()
        if remetente in self._remetentes_bloqueados:
            return self.peso
        return 0.0

    def motivo(self, mensagem: "Mensagem") -> str:
        """Informa se o remetente consta na lista de bloqueio."""
        remetente = mensagem.remetente.strip().casefold()
        if remetente in self._remetentes_bloqueados:
            return f"O remetente {mensagem.remetente!r} consta na lista de bloqueio."
        return "O remetente não consta na lista de bloqueio."
