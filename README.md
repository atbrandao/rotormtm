# RotorMTM - Gyroscopic Metastructure Rotordynamics

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)

Uma biblioteca Python para análise de rotores com metaestruturas giroscópicas, baseada na teoria de ressonadores acoplados para controle de vibrações em sistemas rotativos.

## 🎯 Objetivo

O **RotorMTM** (Rotor MetaStructure) implementa uma abordagem inovadora para o controle de vibrações em sistemas rotativos através da utilização de **metaestruturas giroscópicas**. A biblioteca permite:

- **Análise modal** de rotores com ressonadores acoplados
- **Controle de vibrações** através de atenuação seletiva de frequências
- **Projeto de absorvedores dinâmicos** para máquinas rotativas
- **Análise de resposta forçada** com excitação síncrona e assíncrona
- **Otimização de parâmetros** de ressonadores para máxima eficiência

## 🔬 Fundamentação Teórica

### Metaestruturas Giroscópicas
O conceito baseia-se no acoplamento de **ressonadores giroscópicos** ao rotor principal, criando uma metastrutura capaz de:

1. **Atenuação direcionalmente seletiva**: Controle independente de movimentos forward/backward
2. **Bandgaps de frequência**: Criação de faixas de frequência com baixa transmissibilidade
3. **Efeitos não-recíprocos**: Comportamento dependente da direção de rotação

### Modelo Matemático
O sistema é governado pelas equações:

```
[M]{ẍ} + ([C] + Ω[G]){ẋ} + [K]{x} = {F}
```

Onde:
- `[M]`, `[C]`, `[K]`: Matrizes de massa, amortecimento e rigidez do sistema acoplado
- `[G]`: Matriz giroscópica
- `Ω`: Velocidade de rotação
- `{F}`: Vetor de forças externas

## 📁 Estrutura do Repositório

```
RotorMTM/
├── rotor_mtm_lib/              # Biblioteca principal
│   ├── rotor_mtm/              # Módulos do sistema
│   │   ├── rotor_mtm.py        # Classe principal RotorMTM
│   │   ├── harmbal.py          # Análise harmônica não-linear
│   │   ├── results.py          # Classes de resultados
│   │   └── __init__.py
│   └── setup.py                # Instalação da biblioteca
├── gui/                        # Interfaces gráficas
│   ├── interface_rotor_step1.py # Interface: Configuração do sistema
│   ├── interface_rotor_step2.py # Interface: Análise FRF
│   └── README.md               # Documentação das interfaces
├── examples/                   # Exemplos e casos de estudo
│   ├── app_campbell.py         # Diagramas de Campbell
│   ├── generate_plots.py       # Geração de gráficos
│   └── *.ipynb                # Notebooks Jupyter
├── results/                    # Resultados de análises
├── scripts/                    # Scripts auxiliares
├── tools/                      # Ferramentas de análise
├── Multistage Pump/           # Caso de estudo: Bomba multiestágio
├── Turboexpander/             # Caso de estudo: Turboexpansor
└── README.md                  # Este arquivo
```

## 🚀 Instalação

### Pré-requisitos
```bash
pip install numpy scipy plotly streamlit ross-rotordynamics
```

### Instalação da Biblioteca
```bash
cd rotor_mtm_lib
pip install -e .
```

### Verificação da Instalação
```python
from rotor_mtm.rotor_mtm import RotorMTM
from rotor_mtm.results import LinearResults
print("RotorMTM instalado com sucesso!")
```

## 💡 Exemplo de Uso

### Configuração Básica
```python
import ross as rs
from rotor_mtm.rotor_mtm import RotorMTM
import numpy as np

# Criar rotor base
rotor = rs.rotor_example()

# Configurar ressonadores
n_pos = [5, 10, 15]  # Posições nodais
masses = [1.0, 1.0, 1.0]  # Massas dos ressonadores
Id_values = [1e-3, 1e-3, 1e-3]  # Momentos diametrais
Ip_values = [5e-4, 5e-4, 5e-4]  # Momentos polares

# Criar elementos de disco
resonators = []
for i, pos in enumerate(n_pos):
    disk = rs.DiskElement(
        n=pos, m=masses[i], 
        Id=Id_values[i], Ip=Ip_values[i],
        tag=f'Resonator_{i+1}'
    )
    resonators.append(disk)

# Construir RotorMTM
k0 = 1e6  # Rigidez radial (N/m)
k1 = 1e3  # Rigidez rotacional (N.m/rad)

metarotor = RotorMTM(
    rotor=rotor,
    n_pos=n_pos,
    dk_r=resonators,
    k0=k0,
    k1=k1,
    var=0.1,      # Variação de massa
    var_k=0.1,    # Variação de rigidez
    p_damp=1e-4   # Amortecimento proporcional
)
```

### Análise Modal
```python
# Análise modal para diferentes velocidades
speeds = np.linspace(0, 1000, 51)
modal_results = metarotor.run_analysis(
    sp_arr=speeds,
    n_modes=30,
    diff_analysis=True,
    heatmap=True
)

# Visualizar diagramas de Campbell
from rotor_mtm.rotor_mtm import plot_campbell, plot_diff_modal

fig_campbell = plot_campbell(
    modal_results['w'], 
    speeds
)

fig_diff = plot_diff_modal(
    modal_results['w'],
    modal_results['diff'],
    speeds,
    mode='abs'
)
```

### Análise de Resposta Forçada (FRF)
```python
# Configurar análise FRF
speeds = np.linspace(100, 800, 100)
forces = np.ones_like(speeds)  # Força unitária

# DOFs de interesse
probe_dof = [0, 1, 4, 5]  # x, y dos nós 0 e 1
probe_names = ['Node_0_x', 'Node_0_y', 'Node_1_x', 'Node_1_y']

# Executar análise
linear_results = metarotor.calc_frf(
    sp_arr=speeds,
    f=forces,
    probe_dof=probe_dof,
    probe_names=probe_names,
    f_node=0,  # Nó de excitação
    rotor_solo=False
)

# Plotar FRF
fig_forward, fig_backward = linear_results.plot_frf(
    dof=probe_names,
    whirl='both',
    amplitude_units='rms'
)
```

### Análise Comparativa
```python
# Comparar com rotor sem ressonadores
linear_results_solo = metarotor.calc_frf(
    sp_arr=speeds,
    f=forces,
    probe_dof=probe_dof[:2],  # Apenas DOFs do rotor
    probe_names=probe_names[:2],
    f_node=0,
    rotor_solo=True  # Rotor sem ressonadores
)

# Calcular eficiência de atenuação
efficiency = np.abs(linear_results.rf['Node_0_x']) / np.abs(linear_results_solo.rf['Node_0_x'])
```

## 🖥️ Interface Gráfica

O RotorMTM inclui interfaces gráficas desenvolvidas em Streamlit para facilitar o uso:

### Etapa 1: Configuração do Sistema
```bash
streamlit run gui/interface_rotor_step1.py
```
- Carregamento de rotores (arquivo ou exemplo)
- Configuração interativa de ressonadores
- Visualização do sistema
- Construção e salvamento do RotorMTM

### Etapa 2: Análise FRF
```bash
streamlit run gui/interface_rotor_step2.py
```
- Carregamento de sistemas RotorMTM
- Configuração de análises FRF
- Visualização de resultados
- Análise comparativa com rotor solo

## 📊 Casos de Estudo

### 1. Bomba Multiestágio (`Multistage Pump/`)
- Análise de bomba centrífuga com 5 estágios
- Otimização de ressonadores para controle de instabilidade
- Comparação com dados experimentais

### 2. Turboexpansor (`Turboexpander/`)
- Sistema turboexpansor-compressor
- Controle de vibrações em alta rotação
- Análise de efeitos não-lineares

### 3. Compressor (`Turboexpander Compressor/`)
- Compressor centrífugo industrial
- Absorvedores dinâmicos otimizados
- Validação com dados de campo

## 📈 Funcionalidades Avançadas

### Análise Não-Linear
```python
# Criar sistema não-linear com rigidez cúbica
nonlinear_system = metarotor.create_Sys_NL(
    x_eq0=(0.001, None),  # Equilíbrio radial
    x_eq1=(None, None),   # Equilíbrio rotacional
    sp=500,               # Velocidade de referência
    n_harm=10,            # Harmônicos
    nu=1                  # Inter-harmônicos
)

# Análise harmônica
from rotor_mtm.harmbal import run_integration
nonlinear_results = run_integration(
    system=nonlinear_system,
    frequency_list=np.linspace(400, 600, 21),
    initial_conditions='auto'
)
```

### Rainbow Metastructures
```python
# Configurar variação gradual de propriedades
metarotor_rainbow = RotorMTM(
    rotor=rotor,
    n_pos=n_pos,
    dk_r=resonators,
    k0=k0,
    k1=k1,
    var=0.3,      # Variação de 30% na massa
    var_k=0.2,    # Variação de 20% na rigidez
    exp_var=2     # Variação quadrática
)
```

### Otimização de Parâmetros
```python
# Função objetivo: minimizar resposta em frequência crítica
def objective_function(params):
    k0, k1 = params
    temp_rotor = RotorMTM(rotor, n_pos, resonators, k0, k1)
    results = temp_rotor.calc_frf(critical_speeds, forces, probe_dof)
    return np.max(np.abs(results.rf['Node_0_x']))

# Otimização com scipy
from scipy.optimize import minimize
result = minimize(
    objective_function,
    x0=[1e6, 1e3],
    bounds=[(1e4, 1e8), (1e1, 1e5)],
    method='L-BFGS-B'
)
```

## 📚 Referências Acadêmicas

### Artigos Principais

1. **Brandão, A.T., et al. (2022)**  
   *"Gyroscopic metastructures for vibration control in rotating machinery"*  
   Journal of Sound and Vibration, 516, 116982.  
   DOI: [10.1016/j.jsv.2022.116982](https://doi.org/10.1016/j.jsv.2022.116982)

2. **Brandão, A.T., et al. (2025)**  
   *"Nonlinear dynamics of gyroscopic metastructures with bi-stable resonators"*  
   Nonlinear Dynamics (aceito para publicação)  
   DOI: [10.1007/s11071-025-11597-z](https://doi.org/10.1007/s11071-025-11597-z)

3. **Hussein, M.I., Leamy, M.J., Ruzzene, M. (2014)**  
   *"Dynamics of phononic materials and structures: Historical origins, recent progress, and future outlook"*  
   Applied Mechanics Reviews, 66(4), 040802.

### Teoria de Base

4. **Vangbo, M. (1998)**  
   *"An analytical analysis of a compressed bistable buckled beam"*  
   Sensors and Actuators A: Physical, 69(3), 212-216.

5. **Genta, G. (2005)**  
   *"Dynamics of Rotating Systems"*  
   Springer Science & Business Media.

6. **Lalanne, M., Ferraris, G. (1998)**  
   *"Rotordynamics Prediction in Engineering"*  
   John Wiley & Sons.

### Aplicações e Métodos

7. **Yu, D., et al. (2006)**  
   *"Flexural vibration band gaps in Timoshenko beams with locally resonant structures"*  
   Journal of Applied Physics, 100(12), 124901.

8. **Ma, G., Sheng, P. (2016)**  
   *"Acoustic metamaterials: From local resonances to broad horizons"*  
   Science Advances, 2(2), e1501595.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. **Fork** o repositório
2. **Crie** uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. **Commit** suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. **Push** para a branch (`git push origin feature/nova-feature`)
5. **Abra** um Pull Request

### Diretrizes de Contribuição
- Siga as convenções de código Python (PEP 8)
- Adicione testes para novas funcionalidades
- Mantenha a documentação atualizada
- Inclua exemplos de uso quando apropriado

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autores

- **Alexandre Tércio Brandão** - *Desenvolvimento principal* - [GitHub](https://github.com/atbrandao)
- **Equipe de Rotordinâmica PETROBRAS/UFRJ**

## 📞 Contato

Para dúvidas, sugestões ou colaborações:
- **Email**: [alexandre.brandao@petrobras.com.br]
- **Issues**: [GitHub Issues](https://github.com/atbrandao/rotormtm/issues)

## 🙏 Agradecimentos

- **PETROBRAS** - Apoio institucional e financiamento
- **UFRJ/COPPE** - Infraestrutura de pesquisa
- **ROSS Community** - Base para desenvolvimento rotordinâmico
- **SciPy Community** - Ferramentas de computação científica

---

*Desenvolvido com ❤️ para a comunidade de rotordinâmica*
