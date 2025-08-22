"""
Interface Streamlit - Etapa 1: Carregar Rotor e Adicionar Ressonadores
=====================================================================

Interface simples para:
1. Carregar um rotor (arquivo salvo ou exemplo do ROSS)
2. Visualizar o rotor carregado
3. Adicionar ressonadores com geometria e posição personalizadas
4. Salvar a configuração do sistema rotor + ressonadores
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Adicionar o caminho da biblioteca rotor_mtm
sys.path.append('rotor_mtm_lib')

# Configuração da página
st.set_page_config(
    page_title="Rotor MTM - Etapa 1",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔧 Etapa 1: Carregamento de Rotor e Configuração de Ressonadores")
st.markdown("---")

# Verificar e importar bibliotecas necessárias
@st.cache_data
def check_dependencies():
    """Verifica se as dependências necessárias estão disponíveis."""
    missing_deps = []
    available_deps = []
    
    # Testar ROSS
    try:
        import ross as rs
        available_deps.append("ROSS")
    except ImportError:
        missing_deps.append("ross-rotordynamics")
    
    # Testar rotor_mtm
    try:
        from rotor_mtm_lib.rotor_mtm.rotor_mtm import RotorMTM
        available_deps.append("rotor_mtm")
    except ImportError:
        missing_deps.append("rotor_mtm (local)")
    
    return available_deps, missing_deps

# Verificar dependências
available_deps, missing_deps = check_dependencies()

if missing_deps:
    st.error(f"❌ Dependências não encontradas: {', '.join(missing_deps)}")
    st.info("""
    **Para usar esta interface você precisa:**
    1. Instalar ROSS: `pip install ross-rotordynamics`
    2. Ter a biblioteca rotor_mtm no diretório `rotor_mtm_lib/`
    """)
    st.stop()
else:
    st.success(f"✅ Dependências encontradas: {', '.join(available_deps)}")

# Importar bibliotecas
import ross as rs
from rotor_mtm_lib.rotor_mtm.rotor_mtm import RotorMTM

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def create_2d_system_visualization():
    """Cria visualização 2D do sistema como fallback."""
    fig_system = go.Figure()
    
    # Eixo do rotor
    x = st.session_state.rotor.nodes_pos
    y_top = [0.025] * len(x)
    y_bottom = [-0.025] * len(x)
    
    fig_system.add_trace(go.Scatter(
        x=x, y=y_top, mode='lines', name='Eixo Superior',
        line=dict(color='steelblue', width=4), showlegend=False
    ))
    fig_system.add_trace(go.Scatter(
        x=x, y=y_bottom, mode='lines', name='Eixo Inferior',
        line=dict(color='steelblue', width=4), showlegend=False,
        fill='tonexty', fillcolor='rgba(70,130,180,0.3)'
    ))
    
    # Discos originais
    for disk in st.session_state.rotor.disk_elements:
        pos = st.session_state.rotor.nodes_pos[disk.n]
        fig_system.add_shape(
            type="rect",
            x0=pos-0.02, y0=-0.05, x1=pos+0.02, y1=0.05,
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0.3)"
        )
        
        fig_system.add_annotation(
            x=pos, y=0.08, text=f"Disco",
            showarrow=False, font=dict(size=10, color="red")
        )
    
    # Mancais
    for bearing in st.session_state.rotor.bearing_elements:
        pos = st.session_state.rotor.nodes_pos[bearing.n]
        fig_system.add_shape(
            type="rect",
            x0=pos-0.01, y0=-0.03, x1=pos+0.01, y1=0.03,
            line=dict(color="green", width=2),
            fillcolor="rgba(0,255,0,0.5)"
        )
    
    # Ressonadores
    for res in st.session_state.resonators_config:
        pos = res['position_m']
        
        # Ressonador como círculo conectado por linha
        fig_system.add_shape(
            type="circle",
            x0=pos-0.015, y0=0.08-0.015, x1=pos+0.015, y1=0.08+0.015,
            line=dict(color="orange", width=2),
            fillcolor="rgba(255,165,0,0.6)"
        )
        
        # Linha de conexão
        fig_system.add_shape(
            type="line",
            x0=pos, y0=0.025, x1=pos, y1=0.065,
            line=dict(color="darkorange", width=3, dash="dash")
        )
        
        # Label
        fig_system.add_annotation(
            x=pos, y=0.12, text=f"R{res['id']}<br>{res['f0']:.1f}Hz",
            showarrow=False, font=dict(size=9, color="orange")
        )
    
    fig_system.update_layout(
        title=f"Sistema Rotor + {len(st.session_state.resonators_config)} Ressonadores (Vista 2D)",
        xaxis_title="Posição Axial (m)",
        yaxis_title="Raio (m)",
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig_system

def create_rotor_mtm():
    """Cria ou atualiza o objeto RotorMTM com base nos ressonadores configurados."""
    if st.session_state.rotor is None or not st.session_state.resonators_config:
        st.session_state.rotor_mtm = None
        return False
    
    try:
        # Obter números dos nós dos ressonadores (não as posições axiais)
        # res['position'] = número do nó (0, 1, 2, ...)
        # res['position_m'] = posição axial em metros
        n_pos = [res['position'] for res in st.session_state.resonators_config]
        
        # Criar discos ressonadores individuais com as posições corretas
        dk_r = []
        for i, res in enumerate(st.session_state.resonators_config):
            disk = rs.DiskElement(
                n=res['position'],  # Usar a posição correta do nó
                m=res['mass'],
                Id=res['Id'], 
                Ip=res['Ip'],
                tag=f'resonator_{i}'
            )
            dk_r.append(disk)
        
        # Para simplificar, usar o mesmo k0 e k1 do primeiro ressonador
        # Em versões futuras, isso pode ser individualizado
        if st.session_state.resonators_config:
            k0 = st.session_state.resonators_config[0]['k0']
            k1 = st.session_state.resonators_config[0]['k1']
        else:
            k0, k1 = 1e6, 1e4
        
        # Criar RotorMTM
        st.session_state.rotor_mtm = RotorMTM(
            rotor=st.session_state.rotor,
            n_pos=n_pos,  # Números dos nós
            dk_r=dk_r,
            k0=k0,
            k1=k1,
            var=0,  # Sem variação por enquanto
            var_k=0,
            p_damp=1e-4,
            ge=True  # Efeitos giroscópicos
        )
        
        return True
        
    except Exception as e:
        st.error(f"Erro ao criar RotorMTM: {e}")
        st.session_state.rotor_mtm = None
        return False

# Importar bibliotecas
import ross as rs
from rotor_mtm_lib.rotor_mtm.rotor_mtm import RotorMTM

# Inicializar session state
if 'rotor' not in st.session_state:
    st.session_state.rotor = None
if 'rotor_source' not in st.session_state:
    st.session_state.rotor_source = None
if 'resonators_config' not in st.session_state:
    st.session_state.resonators_config = []
if 'rotor_mtm' not in st.session_state:
    st.session_state.rotor_mtm = None

# ============================================================================
# SIDEBAR: CONFIGURAÇÕES GLOBAIS
# ============================================================================

with st.sidebar:
    st.header("🎛️ Configurações")
    
    st.subheader("Status Atual")
    rotor_status = "✅ Carregado" if st.session_state.rotor is not None else "❌ Não carregado"
    st.write(f"**Rotor:** {rotor_status}")
    st.write(f"**Ressonadores:** {len(st.session_state.resonators_config)}")
    
    if st.session_state.rotor is not None:
        st.write(f"**Fonte:** {st.session_state.rotor_source}")
    
    st.markdown("---")
    
    # Botão para limpar tudo
    if st.button("🗑️ Limpar Tudo", type="secondary"):
        st.session_state.rotor = None
        st.session_state.rotor_source = None
        st.session_state.resonators_config = []
        st.session_state.rotor_mtm = None
        st.rerun()

# ============================================================================
# SEÇÃO 1: CARREGAMENTO DO ROTOR
# ============================================================================

st.header("📁 1. Carregamento do Rotor")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Opções de Carregamento")
    
    load_option = st.radio(
        "Escolha como carregar o rotor:",
        ["Usar exemplo do ROSS", "Carregar arquivo salvo (.pkl)", "Criar novo rotor"]
    )
    
    if load_option == "Usar exemplo do ROSS":
        st.info("Carregando exemplo pré-configurado do ROSS")
        
        if st.button("🔧 Carregar Rotor de Exemplo", type="primary"):
            try:
                # Carregar rotor de exemplo do ROSS
                rotor = rs.rotor_example()
                st.session_state.rotor = rotor
                st.session_state.rotor_source = "Exemplo ROSS"
                st.success("✅ Rotor de exemplo carregado com sucesso!")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao carregar rotor de exemplo: {e}")
                
    elif load_option == "Carregar arquivo salvo (.pkl)":
        uploaded_file = st.file_uploader(
            "Selecione um arquivo de rotor (.pkl)",
            type=['pkl'],
            help="Arquivo pickle contendo um objeto Rotor do ROSS"
        )
        
        if uploaded_file is not None:
            if st.button("📂 Carregar Arquivo", type="primary"):
                try:
                    import pickle
                    rotor = pickle.load(uploaded_file)
                    st.session_state.rotor = rotor
                    st.session_state.rotor_source = f"Arquivo: {uploaded_file.name}"
                    st.success(f"✅ Rotor carregado de {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao carregar arquivo: {e}")
    
    elif load_option == "Criar novo rotor":
        st.info("🚧 Funcionalidade de criação será implementada na próxima etapa")

with col2:
    st.subheader("Informações do Rotor")
    
    if st.session_state.rotor is not None:
        rotor = st.session_state.rotor
        
        # Informações básicas
        rotor_info = {
            "Fonte": st.session_state.rotor_source,
            "Elementos do eixo": len(rotor.shaft_elements),
            "Discos": len(rotor.disk_elements),
            "Mancais": len(rotor.bearing_elements),
            "Graus de liberdade": rotor.ndof,
            "Comprimento total": f"{rotor.nodes_pos[-1]:.3f} m",
            "Nós": len(rotor.nodes_pos)
        }
        
        for key, value in rotor_info.items():
            st.write(f"**{key}:** {value}")
        
        # Tabela de nós
        with st.expander("📍 Posições dos Nós"):
            nodes_df = pd.DataFrame({
                'Nó': range(len(rotor.nodes_pos)),
                'Posição (m)': rotor.nodes_pos
            })
            st.dataframe(nodes_df, use_container_width=True)
            
    else:
        st.info("👈 Carregue um rotor primeiro para ver as informações")

# ============================================================================
# SEÇÃO 2: VISUALIZAÇÃO DO ROTOR
# ============================================================================

if st.session_state.rotor is not None:
    st.header("🔍 2. Visualização do Rotor")
    
    # Opções de visualização
    col_vis1, col_vis2, col_vis3 = st.columns([1, 1, 1])
    
    with col_vis1:
        show_disk_3d = st.checkbox("Mostrar discos em 3D", True)
    with col_vis2:
        show_bearings = st.checkbox("Mostrar mancais", True)
    with col_vis3:
        fig_height = st.slider("Altura do gráfico", 400, 800, 600)
    
    try:
        with st.spinner("Gerando visualização 3D..."):
            # Usar plot_rotor do ROSS
            fig = st.session_state.rotor.plot_rotor()
            
            # Atualizar layout básico
            fig.update_layout(
                title="Visualização 3D do Rotor Carregado",
                height=fig_height,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Erro na visualização: {e}")
        st.info("Tentando visualização alternativa...")
        
        # Visualização simples 2D como fallback
        try:
            fig_2d = go.Figure()
            
            # Eixo
            x = st.session_state.rotor.nodes_pos
            y_top = [0.025] * len(x)
            y_bottom = [-0.025] * len(x)
            
            fig_2d.add_trace(go.Scatter(
                x=x, y=y_top, mode='lines', name='Eixo Superior',
                line=dict(color='steelblue', width=4), showlegend=False
            ))
            fig_2d.add_trace(go.Scatter(
                x=x, y=y_bottom, mode='lines', name='Eixo Inferior',
                line=dict(color='steelblue', width=4), showlegend=False,
                fill='tonexty', fillcolor='rgba(70,130,180,0.3)'
            ))
            
            # Discos
            for disk in st.session_state.rotor.disk_elements:
                pos = st.session_state.rotor.nodes_pos[disk.n]
                fig_2d.add_shape(
                    type="rect",
                    x0=pos-0.02, y0=-0.05, x1=pos+0.02, y1=0.05,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(255,0,0,0.3)"
                )
            
            # Mancais
            for bearing in st.session_state.rotor.bearing_elements:
                pos = st.session_state.rotor.nodes_pos[bearing.n]
                fig_2d.add_shape(
                    type="rect",
                    x0=pos-0.01, y0=-0.03, x1=pos+0.01, y1=0.03,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0,255,0,0.5)"
                )
            
            fig_2d.update_layout(
                title="Visualização 2D do Rotor (Fallback)",
                xaxis_title="Posição Axial (m)",
                yaxis_title="Raio (m)",
                height=fig_height,
                yaxis=dict(scaleanchor="x", scaleratio=1),
                showlegend=False
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
            
        except Exception as e2:
            st.error(f"Erro na visualização alternativa: {e2}")

# ============================================================================
# SEÇÃO 3: CONFIGURAÇÃO DOS RESSONADORES
# ============================================================================

if st.session_state.rotor is not None:
    st.header("🎼 3. Configuração dos Ressonadores")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Adicionar Ressonador")
        
        # Posição do ressonador
        max_node = len(st.session_state.rotor.nodes_pos) - 1
        resonator_position = st.selectbox(
            "Posição do ressonador (nó):",
            range(max_node + 1),
            help=f"Escolha um nó entre 0 e {max_node}"
        )
        
        # Mostrar posição em metros
        pos_meters = st.session_state.rotor.nodes_pos[resonator_position]
        st.info(f"📍 Posição selecionada: {pos_meters:.3f} m")
        
        # Propriedades do disco ressonador
        st.subheader("Propriedades do Disco Ressonador")
        resonator_mass = st.number_input("Massa (kg)", 0.1, 100.0, 5.0, 0.1)
        resonator_Id = st.number_input("Id - Momento de inércia diametral (kg⋅m²)", 0.001, 10.0, 0.1, 0.001)
        resonator_Ip = st.number_input("Ip - Momento de inércia polar (kg⋅m²)", 0.001, 10.0, 0.05, 0.001)
        
        # Propriedades de conexão
        st.subheader("Propriedades de Conexão")
        k0 = st.number_input("Rigidez translacional k0 (N/m)", 1e3, 1e8, 1e6, format="%.0e")
        k1 = st.number_input("Rigidez rotacional k1 (N⋅m/rad)", 1e2, 1e6, 1e4, format="%.0e")
        
        # Calcular frequência de sintonia
        f0 = np.sqrt(k0 / resonator_mass) / (2 * np.pi)
        f1 = np.sqrt(k1 / resonator_Id) / (2 * np.pi)
        
        st.write(f"**Frequência translacional f0:** {f0:.2f} Hz")
        st.write(f"**Frequência rotacional f1:** {f1:.2f} Hz")
        
        # Botão para adicionar ressonador
        if st.button("➕ Adicionar Ressonador", type="primary"):
            # Verificar se já existe ressonador nesta posição
            existing = [r for r in st.session_state.resonators_config if r['position'] == resonator_position]
            
            if existing:
                st.warning(f"⚠️ Já existe um ressonador na posição {resonator_position}")
            else:
                new_resonator = {
                    'id': len(st.session_state.resonators_config) + 1,
                    'position': resonator_position,
                    'position_m': pos_meters,
                    'mass': resonator_mass,
                    'Id': resonator_Id,
                    'Ip': resonator_Ip,
                    'k0': k0,
                    'k1': k1,
                    'f0': f0,
                    'f1': f1
                }
                
                st.session_state.resonators_config.append(new_resonator)
                
                # Criar/atualizar RotorMTM
                if create_rotor_mtm():
                    st.success(f"✅ Ressonador {new_resonator['id']} adicionado na posição {resonator_position}")
                    st.success("🎼 RotorMTM atualizado com novo ressonador")
                else:
                    st.success(f"✅ Ressonador {new_resonator['id']} adicionado na posição {resonator_position}")
                    st.warning("⚠️ RotorMTM não pôde ser criado")
                
                st.rerun()
    
    with col2:
        st.subheader("Ressonadores Configurados")
        
        if st.session_state.resonators_config:
            for i, res in enumerate(st.session_state.resonators_config):
                with st.expander(f"🎼 Ressonador {res['id']} - Nó {res['position']} ({res['position_m']:.3f} m)"):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**Massa:** {res['mass']:.2f} kg")
                        st.write(f"**Id:** {res['Id']:.4f} kg⋅m²")
                        st.write(f"**Ip:** {res['Ip']:.4f} kg⋅m²")
                    
                    with col_info2:
                        st.write(f"**k0:** {res['k0']:.2e} N/m")
                        st.write(f"**k1:** {res['k1']:.2e} N⋅m/rad")
                        st.write(f"**f0:** {res['f0']:.2f} Hz")
                        st.write(f"**f1:** {res['f1']:.2f} Hz")
                    
                    if st.button(f"🗑️ Remover Ressonador {res['id']}", key=f"remove_{res['id']}"):
                        st.session_state.resonators_config = [
                            r for r in st.session_state.resonators_config if r['id'] != res['id']
                        ]
                        
                        # Recriar RotorMTM após remoção
                        create_rotor_mtm()
                        st.success(f"Ressonador {res['id']} removido")
                        st.rerun()
        else:
            st.info("👈 Adicione ressonadores para visualizá-los aqui")
        
        # Tabela resumo
        if st.session_state.resonators_config:
            st.subheader("Resumo dos Ressonadores")
            df_resonators = pd.DataFrame(st.session_state.resonators_config)
            st.dataframe(
                df_resonators[['id', 'position', 'position_m', 'mass', 'f0', 'f1']].round(3),
                use_container_width=True
            )

# ============================================================================
# SEÇÃO 4: VISUALIZAÇÃO COM RESSONADORES
# ============================================================================

if st.session_state.rotor is not None and st.session_state.resonators_config:
    st.header("🔍 4. Visualização do Sistema Rotor + Ressonadores")
    
    # Opções de visualização
    col_vis1, col_vis2 = st.columns([1, 1])
    
    with col_vis1:
        fig_height_mtm = st.slider("Altura do gráfico", 400, 800, 600, key="mtm_height")
    with col_vis2:
        st.info("ℹ️ A visualização 3D será gerada automaticamente pelo RotorMTM")
    
    if st.session_state.rotor_mtm is not None:
        st.subheader("🎼 Visualização 3D usando RotorMTM")
        
        try:
            with st.spinner("Gerando visualização 3D do RotorMTM..."):
                # Usar o método plot_rotor() da classe RotorMTM sem argumentos
                fig_mtm = st.session_state.rotor_mtm.plot_rotor()
                
                # Atualizar layout
                fig_mtm.update_layout(
                    title="Visualização 3D do RotorMTM (Rotor + Ressonadores)",
                    height=fig_height_mtm,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                st.plotly_chart(fig_mtm, use_container_width=True)
                
                # Informações do RotorMTM
                with st.expander("ℹ️ Informações do RotorMTM"):
                    mtm_info = {
                        "Ressonadores": st.session_state.rotor_mtm.n_res,
                        "Graus de liberdade total": st.session_state.rotor_mtm.N,
                        "Graus de liberdade rotor original": st.session_state.rotor_mtm.N2,
                        "Razão de massa": f"{st.session_state.rotor_mtm.m_ratio:.4f}",
                        "Rigidez translacional k0": f"{st.session_state.rotor_mtm.k0:.2e} N/m",
                        "Rigidez rotacional k1": f"{st.session_state.rotor_mtm.k1:.2e} N⋅m/rad"
                    }
                    
                    for key, value in mtm_info.items():
                        st.write(f"**{key}:** {value}")
                
        except Exception as e:
            st.error(f"Erro na visualização 3D do RotorMTM: {e}")
            st.info("Usando visualização 2D alternativa...")
            
            # Fallback para visualização 2D
            fig_fallback = create_2d_system_visualization()
            st.plotly_chart(fig_fallback, use_container_width=True)
    
    else:
        st.warning("⚠️ RotorMTM não foi criado. Usando visualização 2D alternativa.")
        fig_2d = create_2d_system_visualization()
        st.plotly_chart(fig_2d, use_container_width=True)

def create_2d_system_visualization():
    """Cria visualização 2D do sistema como fallback."""
    fig_system = go.Figure()
    
    # Eixo do rotor
    x = st.session_state.rotor.nodes_pos
    y_top = [0.025] * len(x)
    y_bottom = [-0.025] * len(x)
    
    fig_system.add_trace(go.Scatter(
        x=x, y=y_top, mode='lines', name='Eixo Superior',
        line=dict(color='steelblue', width=4), showlegend=False
    ))
    fig_system.add_trace(go.Scatter(
        x=x, y=y_bottom, mode='lines', name='Eixo Inferior',
        line=dict(color='steelblue', width=4), showlegend=False,
        fill='tonexty', fillcolor='rgba(70,130,180,0.3)'
    ))
    
    # Discos originais
    for disk in st.session_state.rotor.disk_elements:
        pos = st.session_state.rotor.nodes_pos[disk.n]
        fig_system.add_shape(
            type="rect",
            x0=pos-0.02, y0=-0.05, x1=pos+0.02, y1=0.05,
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0.3)"
        )
        
        fig_system.add_annotation(
            x=pos, y=0.08, text=f"Disco",
            showarrow=False, font=dict(size=10, color="red")
        )
    
    # Mancais
    for bearing in st.session_state.rotor.bearing_elements:
        pos = st.session_state.rotor.nodes_pos[bearing.n]
        fig_system.add_shape(
            type="rect",
            x0=pos-0.01, y0=-0.03, x1=pos+0.01, y1=0.03,
            line=dict(color="green", width=2),
            fillcolor="rgba(0,255,0,0.5)"
        )
    
    # Ressonadores
    for res in st.session_state.resonators_config:
        pos = res['position_m']
        
        # Ressonador como círculo conectado por linha
        fig_system.add_shape(
            type="circle",
            x0=pos-0.015, y0=0.08-0.015, x1=pos+0.015, y1=0.08+0.015,
            line=dict(color="orange", width=2),
            fillcolor="rgba(255,165,0,0.6)"
        )
        
        # Linha de conexão
        fig_system.add_shape(
            type="line",
            x0=pos, y0=0.025, x1=pos, y1=0.065,
            line=dict(color="darkorange", width=3, dash="dash")
        )
        
        # Label
        fig_system.add_annotation(
            x=pos, y=0.12, text=f"R{res['id']}<br>{res['f0']:.1f}Hz",
            showarrow=False, font=dict(size=9, color="orange")
        )
    
    return fig_system

# ============================================================================
# SEÇÃO 5: SALVAR CONFIGURAÇÃO
# ============================================================================

if st.session_state.rotor is not None:
    st.header("💾 5. Salvar Configuração")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Salvar Sistema Completo")
        
        filename = st.text_input(
            "Nome do arquivo:",
            value=f"rotor_system_{len(st.session_state.resonators_config)}res",
            help="Nome do arquivo sem extensão"
        )
        
        if st.button("💾 Salvar Configuração", type="primary"):
            try:
                import pickle
                import datetime
                
                # Criar dados para salvar
                save_data = {
                    'rotor': st.session_state.rotor,
                    'rotor_source': st.session_state.rotor_source,
                    'resonators_config': st.session_state.resonators_config,
                    'rotor_mtm': st.session_state.rotor_mtm,  # Incluir objeto RotorMTM
                    'timestamp': datetime.datetime.now(),
                    'n_resonators': len(st.session_state.resonators_config),
                    'version': '1.0'
                }
                
                # Salvar arquivo
                filename_full = f"{filename}.pkl"
                with open(filename_full, 'wb') as f:
                    pickle.dump(save_data, f)
                
                st.success(f"✅ Configuração salva como {filename_full}")
                
                # Mostrar informações do arquivo salvo
                st.info(f"""
                **Arquivo salvo:** {filename_full}
                **Rotor:** {st.session_state.rotor_source}
                **Ressonadores:** {len(st.session_state.resonators_config)}
                **Data:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
            except Exception as e:
                st.error(f"Erro ao salvar: {e}")
        
        # Botão adicional para salvar apenas o RotorMTM (para Etapa 2)
        st.markdown("---")
        
        filename_mtm = st.text_input(
            "Nome para RotorMTM:",
            value=f"rotor_mtm_{len(st.session_state.resonators_config)}res",
            help="Nome do arquivo apenas com o objeto RotorMTM para usar na Etapa 2"
        )
        
        if st.button("🔧 Salvar RotorMTM para Etapa 2", type="secondary"):
            if st.session_state.rotor_mtm is not None:
                try:
                    import pickle
                    
                    filename_mtm_full = f"{filename_mtm}.pkl"
                    with open(filename_mtm_full, 'wb') as f:
                        pickle.dump(st.session_state.rotor_mtm, f)
                    
                    st.success(f"✅ RotorMTM salvo como {filename_mtm_full}")
                    st.info("🚀 Este arquivo pode ser usado diretamente na Etapa 2 para análise FRF")
                    
                except Exception as e:
                    st.error(f"Erro ao salvar RotorMTM: {e}")
            else:
                st.error("❌ Nenhum RotorMTM foi construído ainda. Configure os ressonadores primeiro.")
    
    with col2:
        st.subheader("Resumo do Sistema")
        
        if st.session_state.rotor is not None:
            total_mass_resonators = sum([r['mass'] for r in st.session_state.resonators_config])
            
            summary = {
                "Rotor carregado": st.session_state.rotor_source,
                "Elementos do eixo": len(st.session_state.rotor.shaft_elements),
                "Discos originais": len(st.session_state.rotor.disk_elements),
                "Ressonadores": len(st.session_state.resonators_config),
                "Massa total ressonadores": f"{total_mass_resonators:.2f} kg",
                "Graus de liberdade originais": st.session_state.rotor.ndof,
                "Comprimento do rotor": f"{st.session_state.rotor.nodes_pos[-1]:.3f} m"
            }
            
            for key, value in summary.items():
                st.write(f"**{key}:** {value}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🔧 Interface Rotor MTM - Etapa 1 | 
    Carregar Rotor e Configurar Ressonadores | 
    Baseado em ROSS e rotor_mtm
    </div>
    """, 
    unsafe_allow_html=True
)

# Informações de debug no sidebar
with st.sidebar:
    if st.checkbox("🔍 Debug Info"):
        st.write("**Session State:**")
        st.write(f"- Rotor: {type(st.session_state.rotor).__name__ if st.session_state.rotor else 'None'}")
        st.write(f"- Ressonadores: {len(st.session_state.resonators_config)}")
        st.write(f"- Fonte: {st.session_state.rotor_source}")
