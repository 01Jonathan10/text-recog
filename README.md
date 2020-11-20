# Projeto de Reconhecimento de Texto

Projeto feito pelos Alunos Arthur e Jonathan, para matéria de IC, com a proposta de conseguir reconhecer numa imagem qual o texto dela.

1- Reconhecer onde tem texto.
2- Separar os Caracteres.
3- Reconhecer o Caracter.

Tecnológias utilizadas:

Python 3.8 : [caso precise trocar, atualizar, aqui explica bem ](https://tech.serhatteker.com/post/2019-12/upgrade-python38-on-ubuntu/)

- verificar se é versão 64, entrar no python pelo cmd, bash,
`import struct`
`print (struct.calcsize("P") * 8)`

Instalar o PIP p/ manter atualizado e conseguir pegar as mesmas blibiotecas.

-Faça o download do arquivo get-pip.py fornecido por https://pip.pypa.io usando o seguinte comando:
`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`

-Execute get-pip.py usando o seguinte comando:
`sudo python get-pip.py`

-Após a instalação, execute este comando para verificar se o pip está instalado.
`pip --version`

-Remova o arquivo get-pip.py após instalar o pip.
`rm get-pip.py`

Pandas : `pip install -r requirements.txt`