# Interface RotorMTM - Pasta GUI

Esta pasta contém as interfaces gráficas desenvolvidas em Streamlit para trabalhar com o sistema RotorMTM.

## Arquivos Disponíveis

### Interfaces Funcionais
- **`interface_rotor_step1.py`** - Interface da Etapa 1: Carregamento de rotor e configuração de ressonadores
- **`interface_rotor_step2.py`** - Interface da Etapa 2: Análise FRF e visualização de resultados

### Arquivos de Dados
- **`rotor_mtm_3res.pkl`** - Arquivo exemplo com RotorMTM configurado com 3 ressonadores
- **`rotor_system_3res.pkl`** - Arquivo exemplo adicional para testes

## Como Usar

### Estrutura da Pasta
```
gui/
├── interface_rotor_step1.py    # Interface Etapa 1 (funcionando)
├── interface_rotor_step2.py    # Interface Etapa 2 (funcionando)  
├── README.md                   # Esta documentação
├── rotor_mtm_3res.pkl         # Exemplo RotorMTM (3 ressonadores)
└── rotor_system_3res.pkl      # Exemplo adicional para testes
```

### Pré-requisitos
Certifique-se de que a biblioteca `rotor_mtm` está instalada e acessível:
```python
from rotor_mtm.rotor_mtm import RotorMTM
from rotor_mtm.results import LinearResults
```

### Executar as Interfaces

**Etapa 1 - Configuração do Sistema:**
```bash
streamlit run gui/interface_rotor_step1.py
```

**Etapa 2 - Análise FRF:**
```bash
streamlit run gui/interface_rotor_step2.py
```

## Fluxo de Trabalho

### Etapa 1: Configuração
1. **Carregamento do Rotor**
   - Usar exemplo do ROSS
   - Carregar de arquivo pickle
   
2. **Configuração dos Ressonadores**
   - Definir posições no rotor
   - Configurar propriedades (massa, momentos de inércia)
   
3. **Construção do RotorMTM**
   - Definir parâmetros k0 e k1
   - Construir objeto RotorMTM
   - Salvar arquivo .pkl para uso na Etapa 2

### Etapa 2: Análise FRF
1. **Carregamento do RotorMTM**
   - Carregar arquivo .pkl da Etapa 1
   
2. **Configuração da Análise**
   - Definir faixa de velocidades
   - Configurar parâmetros de excitação
   - Selecionar DOFs para análise
   
3. **Execução e Visualização**
   - Executar análise FRF
   - Visualizar resultados (FRF, órbitas)
   - Comparar com rotor solo
   - Salvar resultados

## Funcionalidades Principais

### Interface Etapa 1
- ✅ Carregamento de rotores (exemplo/arquivo)
- ✅ Configuração interativa de ressonadores
- ✅ Visualização 2D do sistema
- ✅ Construção do RotorMTM
- ✅ Salvamento de configurações

### Interface Etapa 2
- ✅ Carregamento de arquivos RotorMTM
- ✅ Configuração flexível de análise FRF
- ✅ Execução paralela (com/sem ressonadores)
- ✅ Visualização avançada (FRF, órbitas)
- ✅ Salvamento de resultados LinearResults

## Troubleshooting

### Erro de Importação
Se houver erro ao importar `rotor_mtm`, verifique:
- Biblioteca está instalada: `pip install rotor_mtm`
- Biblioteca está no PYTHONPATH
- Versão da biblioteca é compatível

### Arquivo não Reconhecido
Se a Etapa 2 não reconhecer o arquivo da Etapa 1:
- Verifique se o arquivo foi gerado corretamente pela `interface_rotor_step1.py`
- Verifique se o arquivo contém objeto RotorMTM válido
- Teste com arquivo exemplo: `rotor_mtm_3res.pkl`

## Exemplos de Uso

### Arquivo Teste Disponível
Utilize o arquivo `rotor_mtm_3res.pkl` como exemplo para testar a Etapa 2 sem precisar configurar um sistema completo.

### Configurações Típicas
- **k0**: 1e6 N/m (rigidez radial)
- **k1**: 1e3 N.m/rad (rigidez rotacional) 
- **Faixa de velocidades**: 0-1000 rad/s
- **Pontos de análise**: 100 pontos
- **Força de excitação**: 1 N

## Suporte

Para problemas ou dúvidas:
1. Verifique os pré-requisitos de instalação
2. Consulte os logs de erro no terminal
3. Teste com arquivos exemplo fornecidos
4. Verifique a compatibilidade das versões das bibliotecas
