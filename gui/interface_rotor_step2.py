import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import os
from pathlib import Path
import sys

# Configuração da página
st.set_page_config(
    page_title="RotorMTM - Análise FRF",
    page_icon="🔧",
    layout="wide"
)

try:
    from rotor_mtm.rotor_mtm import RotorMTM
    from rotor_mtm.results import LinearResults
    ROTOR_MTM_AVAILABLE = True
except ImportError as e:
    st.error(f"Erro ao importar RotorMTM: {e}")
    ROTOR_MTM_AVAILABLE = False

def load_rotor_mtm(file_path):
    """Carrega um objeto RotorMTM salvo"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Verificar se é um arquivo da etapa 1 (dicionário com dados de configuração)
        if isinstance(data, dict):
            # Verificar se tem o objeto RotorMTM diretamente
            if 'rotor_mtm' in data and data['rotor_mtm'] is not None:
                rotor_mtm = data['rotor_mtm']
                if hasattr(rotor_mtm, 'calc_frf'):
                    st.success(f"✅ RotorMTM carregado do arquivo de configuração")
                    st.info(f"**Arquivo gerado em:** {data.get('timestamp', 'N/A')}")
                    return rotor_mtm
            
            # Se não tem o objeto RotorMTM, mas tem os dados para construir
            elif 'rotor' in data and 'resonators_config' in data:
                st.warning("⚠️ Arquivo contém configuração, mas não o objeto RotorMTM construído")
                st.info("Use a Etapa 1 para reconstruir o sistema com ressonadores")
                return None
            else:
                st.error("Arquivo não contém dados válidos de RotorMTM")
                return None
        
        # Verificar se é um objeto RotorMTM diretamente
        elif hasattr(data, 'calc_frf'):
            st.success("✅ Objeto RotorMTM carregado diretamente")
            return data
        else:
            st.error("Arquivo não contém um objeto RotorMTM válido")
            return None
            
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

def save_linear_results(results, filename):
    """Salva o objeto LinearResults"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        return True
    except Exception as e:
        st.error(f"Erro ao salvar resultados: {e}")
        return False

def create_frf_analysis_interface():
    """Interface principal para análise FRF"""
    
    st.title("🔧 RotorMTM - Análise de FRF")
    st.markdown("---")
    
    if not ROTOR_MTM_AVAILABLE:
        st.error("RotorMTM não disponível. Verifique a instalação da biblioteca.")
        return
    
    # Sidebar para configurações
    st.sidebar.header("⚙️ Configurações")
    
    # Seção 1: Carregar RotorMTM
    st.header("1. Carregar RotorMTM")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        rotor_file = st.file_uploader(
            "Selecione o arquivo RotorMTM (.pkl)",
            type=['pkl'],
            help="Arquivo gerado na Etapa 1 da interface"
        )
    
    with col2:
        if rotor_file is not None:
            st.info("✅ Arquivo carregado")
        else:
            st.warning("⚠️ Nenhum arquivo selecionado")
    
    # Carregar RotorMTM
    rotor_mtm = None
    if rotor_file is not None:
        # Salvar arquivo temporariamente
        temp_path = f"temp_{rotor_file.name}"
        with open(temp_path, "wb") as f:
            f.write(rotor_file.getbuffer())
        
        rotor_mtm = load_rotor_mtm(temp_path)
        
        # Remover arquivo temporário
        try:
            os.remove(temp_path)
        except:
            pass
    
    if rotor_mtm is None:
        st.info("📁 Carregue um arquivo RotorMTM para continuar")
        return
    
    # Exibir informações do sistema carregado
    st.success(f"✅ RotorMTM carregado com sucesso!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Número de ressonadores", rotor_mtm.n_res)
    with col2:
        st.metric("Posições dos ressonadores", len(rotor_mtm.n_pos))
    with col3:
        st.metric("Rigidez k0", f"{rotor_mtm.k0:.2e}")
    with col4:
        st.metric("Rigidez k1", f"{rotor_mtm.k1:.2e}")
    
    st.markdown("---")
    
    # Seção 2: Configurar Análise FRF
    st.header("2. Configurar Análise FRF")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parâmetros de Velocidade")
        
        # Velocidades de análise
        speed_min = st.number_input(
            "Velocidade mínima (rad/s)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            help="Velocidade mínima para análise FRF"
        )
        
        speed_max = st.number_input(
            "Velocidade máxima (rad/s)",
            min_value=speed_min + 1,
            value=1000.0,
            step=10.0,
            help="Velocidade máxima para análise FRF"
        )
        
        n_speeds = st.number_input(
            "Número de pontos de velocidade",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Número de pontos entre velocidade mínima e máxima"
        )
        
    with col2:
        st.subheader("Parâmetros de Excitação")
        
        # Força de excitação
        force_magnitude = st.number_input(
            "Magnitude da força (N)",
            min_value=0.01,
            value=1.0,
            step=0.1,
            help="Magnitude da força de excitação harmônica"
        )
        
        # Nó de excitação
        max_node = rotor_mtm.rotor_solo.nodes_pos[-1] if hasattr(rotor_mtm.rotor_solo, 'nodes_pos') else 10
        excitation_node = st.number_input(
            "Nó de excitação",
            min_value=0,
            max_value=int(max_node),
            value=0,
            step=1,
            help="Nó onde a força de excitação é aplicada"
        )
        
        # Comparar com rotor solo
        compare_solo = st.checkbox(
            "Comparar com rotor solo",
            value=True,
            help="Incluir análise do rotor sem ressonadores para comparação"
        )
    
    # Configurações dos DOFs para análise
    st.subheader("DOFs de Análise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # DOFs de sondagem
        probe_nodes = st.multiselect(
            "Nós de sondagem",
            options=list(range(int(max_node) + 1)),
            default=[0],
            help="Nós onde a resposta será medida"
        )
        
    with col2:
        # Tipos de DOF
        dof_types = st.multiselect(
            "Tipos de DOF",
            options=['x', 'y', 'theta_x', 'theta_y'],
            default=['x', 'y'],
            help="Graus de liberdade para análise"
        )
    
    st.markdown("---")
    
    # Seção 3: Executar Análise
    st.header("3. Executar Análise FRF")
    
    if st.button("🚀 Executar Análise FRF", type="primary"):
        
        if len(probe_nodes) == 0 or len(dof_types) == 0:
            st.error("Selecione pelo menos um nó de sondagem e um tipo de DOF")
            return
        
        # Criar array de velocidades
        speed_array = np.linspace(speed_min, speed_max, n_speeds)
        
        # Criar array de forças (constante para todas as velocidades)
        force_array = np.ones_like(speed_array) * force_magnitude
        
        # Criar lista de DOFs de sondagem
        probe_dof = []
        probe_names = []
        for node in probe_nodes:
            for i, dof_type in enumerate(['x', 'y', 'theta_x', 'theta_y']):
                if dof_type in dof_types:
                    probe_dof.append(4 * node + i)
                    probe_names.append(f"Node_{node}_{dof_type}")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Executar análise FRF
            status_text.text("Executando análise FRF...")
            progress_bar.progress(0.3)
            
            linear_results = rotor_mtm.calc_frf(
                sp_arr=speed_array,
                f=force_array,
                probe_dof=probe_dof,
                probe_names=probe_names,
                f_node=excitation_node,
                rotor_solo=False,
                silent=True
            )
            
            progress_bar.progress(0.8)
            
            # Executar análise para rotor solo se solicitado
            linear_results_solo = None
            if compare_solo:
                status_text.text("Executando análise para rotor solo...")
                linear_results_solo = rotor_mtm.calc_frf(
                    sp_arr=speed_array,
                    f=force_array,
                    probe_dof=probe_dof[:len(probe_dof)//2] if len(probe_dof) > rotor_mtm.N2//4 else probe_dof,
                    probe_names=probe_names[:len(probe_names)//2] if len(probe_names) > rotor_mtm.N2//4 else probe_names,
                    f_node=excitation_node,
                    rotor_solo=True,
                    silent=True
                )
            
            progress_bar.progress(1.0)
            status_text.text("Análise concluída!")
            
            # Armazenar resultados no session state
            st.session_state['linear_results'] = linear_results
            st.session_state['linear_results_solo'] = linear_results_solo
            st.session_state['analysis_params'] = {
                'speed_array': speed_array,
                'force_array': force_array,
                'probe_nodes': probe_nodes,
                'dof_types': dof_types,
                'excitation_node': excitation_node,
                'compare_solo': compare_solo
            }
            
            st.success("✅ Análise FRF concluída com sucesso!")
            
        except Exception as e:
            st.error(f"Erro durante a análise: {e}")
            progress_bar.empty()
            status_text.empty()
            return
        
        progress_bar.empty()
        status_text.empty()
    
    # Seção 4: Visualizar Resultados
    if 'linear_results' in st.session_state:
        st.markdown("---")
        st.header("4. Resultados da Análise")
        
        linear_results = st.session_state['linear_results']
        linear_results_solo = st.session_state.get('linear_results_solo')
        analysis_params = st.session_state['analysis_params']
        
        # Informações gerais dos resultados
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pontos de frequência", len(linear_results.fl))
        with col2:
            st.metric("DOFs analisados", len(linear_results.rf.keys()))
        with col3:
            st.metric("Faixa de velocidade", f"{analysis_params['speed_array'][0]:.0f} - {analysis_params['speed_array'][-1]:.0f} rad/s")
        
        # Tabs para diferentes visualizações
        tab1, tab2, tab3 = st.tabs(["📈 FRF", "🎯 Órbitas", "💾 Salvar"])
        
        with tab1:
            st.subheader("Função de Resposta em Frequência (FRF)")
            
            # Configurações do plot FRF
            col1, col2 = st.columns(2)
            
            with col1:
                whirl_option = st.selectbox(
                    "Tipo de movimento",
                    options=['both', 'forward', 'backward'],
                    index=0,
                    help="Tipo de movimento para análise"
                )
                
                amplitude_units = st.selectbox(
                    "Unidades de amplitude",
                    options=['rms', 'pk', 'pk-pk'],
                    index=0,
                    help="Unidades para amplitude da resposta"
                )
                
            with col2:
                frequency_units = st.selectbox(
                    "Unidades de frequência",
                    options=['rad/s', 'RPM'],
                    index=0,
                    help="Unidades para eixo de frequência"
                )
                
                selected_dofs = st.multiselect(
                    "DOFs para plotar",
                    options=list(linear_results.rf.keys()),
                    default=list(linear_results.rf.keys())[:3],
                    help="DOFs para incluir no gráfico FRF"
                )
            
            if len(selected_dofs) > 0:
                try:
                    # Gerar gráfico FRF
                    if whirl_option == 'both':
                        fig_forward, fig_backward = linear_results.plot_frf(
                            dof=selected_dofs,
                            whirl=whirl_option,
                            amplitude_units=amplitude_units,
                            frequency_units=frequency_units
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_forward, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_backward, use_container_width=True)
                            
                    else:
                        fig = linear_results.plot_frf(
                            dof=selected_dofs,
                            whirl=whirl_option,
                            amplitude_units=amplitude_units,
                            frequency_units=frequency_units
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Comparação com rotor solo se disponível
                    if linear_results_solo is not None and st.checkbox("Mostrar comparação com rotor solo"):
                        st.subheader("Comparação: Rotor com Ressonadores vs Rotor Solo")
                        
                        # Selecionar DOFs compatíveis
                        compatible_dofs = [dof for dof in selected_dofs if dof in linear_results_solo.rf.keys()]
                        
                        if len(compatible_dofs) > 0:
                            if whirl_option == 'both':
                                fig_solo_forward, fig_solo_backward = linear_results_solo.plot_frf(
                                    dof=compatible_dofs,
                                    whirl=whirl_option,
                                    amplitude_units=amplitude_units,
                                    frequency_units=frequency_units
                                )
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.plotly_chart(fig_solo_forward, use_container_width=True)
                                with col2:
                                    st.plotly_chart(fig_solo_backward, use_container_width=True)
                            else:
                                fig_solo = linear_results_solo.plot_frf(
                                    dof=compatible_dofs,
                                    whirl=whirl_option,
                                    amplitude_units=amplitude_units,
                                    frequency_units=frequency_units
                                )
                                st.plotly_chart(fig_solo, use_container_width=True)
                        else:
                            st.warning("Nenhum DOF compatível encontrado para comparação")
                    
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico FRF: {e}")
            else:
                st.warning("Selecione pelo menos um DOF para plotar")
        
        with tab2:
            st.subheader("Análise de Órbitas")
            
            # Configurações para órbitas
            col1, col2 = st.columns(2)
            
            with col1:
                orbit_frequency = st.number_input(
                    "Frequência para órbita (rad/s)",
                    min_value=float(analysis_params['speed_array'][0]),
                    max_value=float(analysis_params['speed_array'][-1]),
                    value=float(analysis_params['speed_array'][len(analysis_params['speed_array'])//2]),
                    step=10.0,
                    help="Frequência específica para análise de órbita"
                )
                
                orbit_whirl = st.selectbox(
                    "Tipo de movimento (órbita)",
                    options=['forward', 'backward'],
                    index=0,
                    help="Tipo de movimento para órbita"
                )
                
            with col2:
                # Criar pares de DOF para órbitas (x, y)
                available_nodes = list(set([name.split('_')[1] for name in linear_results.rf.keys() if '_x' in name or '_y' in name]))
                
                orbit_nodes = st.multiselect(
                    "Nós para órbitas",
                    options=available_nodes,
                    default=available_nodes[:2] if len(available_nodes) >= 2 else available_nodes,
                    help="Nós para plotar órbitas (x vs y)"
                )
                
                force_scale = st.number_input(
                    "Escala da força",
                    min_value=0.01,
                    value=1.0,
                    step=0.1,
                    help="Fator de escala para visualização da órbita"
                )
            
            if len(orbit_nodes) > 0:
                try:
                    # Criar pares de DOF para órbitas
                    orbit_dofs = []
                    for node in orbit_nodes:
                        x_dof = f"Node_{node}_x"
                        y_dof = f"Node_{node}_y"
                        if x_dof in linear_results.rf.keys() and y_dof in linear_results.rf.keys():
                            orbit_dofs.append((x_dof, y_dof))
                    
                    if len(orbit_dofs) > 0:
                        # Gerar gráfico de órbita
                        fig_orbit = linear_results.plot_orbit(
                            frequency=orbit_frequency,
                            dof=orbit_dofs,
                            whirl=orbit_whirl,
                            f=force_scale
                        )
                        
                        st.plotly_chart(fig_orbit, use_container_width=True)
                        
                        # Mostrar informações da órbita
                        st.info(f"Órbita na frequência {orbit_frequency:.1f} rad/s - Movimento {orbit_whirl}")
                        
                    else:
                        st.warning("Nenhum par de DOF (x,y) válido encontrado para órbitas")
                        
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico de órbita: {e}")
            else:
                st.warning("Selecione pelo menos um nó para análise de órbita")
        
        with tab3:
            st.subheader("Salvar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Resultados disponíveis para salvamento:**")
                st.write("- Objeto LinearResults completo")
                st.write("- Resultados do rotor solo (se calculado)")
                st.write("- Parâmetros da análise")
                
                # Nome do arquivo
                save_filename = st.text_input(
                    "Nome do arquivo",
                    value="linear_results.pkl",
                    help="Nome do arquivo para salvar os resultados"
                )
                
            with col2:
                st.write("**Informações dos resultados:**")
                st.write(f"- Frequências: {len(linear_results.fl)} pontos")
                st.write(f"- DOFs: {len(linear_results.rf.keys())} graus de liberdade")
                st.write(f"- Excitação: Nó {analysis_params['excitation_node']}")
                st.write(f"- Força: {analysis_params['force_array'][0]} N")
            
            if st.button("💾 Salvar LinearResults"):
                if save_filename:
                    try:
                        # Criar dicionário com todos os resultados
                        results_dict = {
                            'linear_results': linear_results,
                            'linear_results_solo': linear_results_solo,
                            'analysis_params': analysis_params
                        }
                        
                        if save_linear_results(results_dict, save_filename):
                            st.success(f"✅ Resultados salvos em '{save_filename}'")
                            
                            # Criar botão de download
                            with open(save_filename, 'rb') as f:
                                st.download_button(
                                    label="📥 Download do arquivo",
                                    data=f.read(),
                                    file_name=save_filename,
                                    mime="application/octet-stream"
                                )
                        else:
                            st.error("Erro ao salvar resultados")
                    except Exception as e:
                        st.error(f"Erro ao salvar: {e}")
                else:
                    st.warning("Digite um nome para o arquivo")

def main():
    """Função principal"""
    create_frf_analysis_interface()

if __name__ == "__main__":
    main()
