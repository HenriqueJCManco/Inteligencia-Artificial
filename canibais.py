from collections import deque


MOVIMENTOS_POSSIVEIS = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]


class Estado:
    def __init__(self, m_esq, c_esq, barco, m_dir, c_dir):
        self.m_esq = m_esq
        self.c_esq = c_esq
        self.barco = barco  # 0 = esquerda, 1 = direita
        self.m_dir = m_dir
        self.c_dir = c_dir
        self.pai = None

    def eh_valido(self):
        lados = [(self.m_esq, self.c_esq), (self.m_dir, self.c_dir)]

        for m, c in lados:
            if m < 0 or c < 0:
                return False
            if m > 0 and m < c:
                return False

        return True

    def eh_final(self):
        return self.m_dir == 3 and self.c_dir == 3

    def lado_do_barco(self):
        return "Esquerda" if self.barco == 0 else "Direita"

    def __eq__(self, outro):
        return (self.m_esq, self.c_esq, self.barco) == (outro.m_esq, outro.c_esq, outro.barco)

    def __hash__(self):
        return hash((self.m_esq, self.c_esq, self.barco))

    def __repr__(self):
        return f"Esq: {self.m_esq}M/{self.c_esq}C | Barco: {self.lado_do_barco()} | Dir: {self.m_dir}M/{self.c_dir}C"


def gerar_sucessores(estado):
    sucessores = []
    indo_pra_direita = estado.barco == 0

    for m, c in MOVIMENTOS_POSSIVEIS:
        if indo_pra_direita:
            novo = Estado(
                estado.m_esq - m, estado.c_esq - c, 1,
                estado.m_dir + m, estado.c_dir + c
            )
        else:
            novo = Estado(
                estado.m_esq + m, estado.c_esq + c, 0,
                estado.m_dir - m, estado.c_dir - c
            )

        if novo.eh_valido():
            novo.pai = estado
            sucessores.append(novo)

    return sucessores


def bfs():
    inicio = Estado(3, 3, 0, 0, 0)
    fila = deque([inicio])
    visitados = {inicio}

    while fila:
        atual = fila.popleft()

        for proximo in gerar_sucessores(atual):
            if proximo in visitados:
                continue

            if proximo.eh_final():
                return proximo

            visitados.add(proximo)
            fila.append(proximo)

    return None


def reconstruir_caminho(solucao):
    caminho = []
    no = solucao
    while no:
        caminho.append(no)
        no = no.pai
    return list(reversed(caminho))


def imprimir_solucao(solucao):
    caminho = reconstruir_caminho(solucao)
    print(f"Solução encontrada em {len(caminho) - 1} movimentos:\n")
    for i, passo in enumerate(caminho):
        print(f"  [{i}] {passo}")


if __name__ == "__main__":
    resultado = bfs()

    if resultado:
        imprimir_solucao(resultado)
    else:
        print("Sem solução.")