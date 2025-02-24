import streamlit as st
import requests
import os, json
from huggingface_hub import model_info


# App title
st.set_page_config(page_title="🚀💻 FlexLLM Server", layout="wide")

# FastAPI server URL
CHAT_URL = "http://localhost:8080/chat/completions/"  # Adjust the port if necessary
FINETUNE_URL = "http://localhost:8080/finetuning/"
REGISTER_ADAPTER_URL = "http://localhost:8080/register_adapter/"

# Initialize session state variables
if 'added_adapters' not in st.session_state:
    st.session_state.added_adapters = []

if 'adapter' not in st.session_state:
    st.session_state.adapter = None
    st.session_state.peft_model_id = None

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def check_model_availability(model_name):
    try:
        info = model_info(model_name)
        return True
    except Exception:
        return False

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function for generating LLaMA2 response
def generate_llama3_response(prompt_input):
    system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are positive in nature."
    
    # Send request to FastAPI server
    response = requests.post(
        CHAT_URL,
        json={
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "decoding_method": decoding_method,
            "peft_model_id": st.session_state.peft_model_id if st.session_state.peft_model_id else None,
            "messages": [{"role": "system", "content": system_prompt}] + st.session_state.messages + [{"role": "user", "content": prompt_input}]
        }
    )

    result = response.json()
    if response.status_code == 200:
        if result["status"] == "success":
            return result["response"]
    else:
        return f"{result['detail']}"

# Sidebar
with st.sidebar:
    st.title('🚀 FlexLLM Server')
    page = st.radio("Choose a page", ["Chat", "Finetune"])
    peft_model_id = None
    if page == "Chat":
        st.header('🦙 Llama Chatbot')
        # st.success('Using local FastAPI server', icon='✅')
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

        st.subheader('Generation parameters')
        max_length = st.sidebar.slider('Max generation length', min_value=64, max_value=2048, value=1024, step=8)
        # selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'], key='selected_model')
        decoding_method = st.sidebar.selectbox('Decoding method', ['Greedy decoding (default)', 'Sampling'], key='decoding_method')
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01, disabled=decoding_method == 'Greedy decoding (default)')
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, disabled=decoding_method == 'Greedy decoding (default)')

        # Single LoRA Adapter
        st.subheader("LoRA Adapter (optional)")
        placeholder_text = "Enter the Huggingface PEFT model ID" if not st.session_state.get("adapter") else f"Currently registered: {st.session_state.adapter}"
        peft_model_name = st.text_input("Add a LoRA Adapter", placeholder=placeholder_text)
        # Button to register the adapter
        if st.button("Register Adapter"):
            if peft_model_name:
                with st.spinner("Registering LoRA adapter..."):
                    response = requests.post(REGISTER_ADAPTER_URL, params={"peft_model_name": peft_model_name})

                result = response.json()
                if response.status_code == 200:
                    if result["status"] == "success":
                        st.session_state.adapter = peft_model_name
                        st.session_state.peft_model_id = result["peft_model_id"]
                        st.success(f"Successfully registered adapter: {peft_model_name}")
                    else:
                        st.error(f"Failed to register adapter: {result['detail']}")
                else:
                    st.error(f"Failed to register adapter: {result['detail']}")
            else:
                st.warning("Please enter a PEFT model ID.")
                
        # Button to remove current adapter
        if st.button("Remove Current Adapters"):
            st.session_state.adapter = None
            st.session_state.peft_model_id = None
            st.success("Adapter has been removed.")

        # Display current adapter
        st.markdown("**Current Adapter:**")
        if st.session_state.adapter:
            st.write(f"- {st.session_state.adapter}")
        else:
            st.write("No adapter registered.")

        # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
    elif page == "Finetune":
        st.header("🏋️‍♂️ LoRA Finetuning")
        
        # Hugging Face token input
        # hf_token = st.text_input("Enter your Hugging Face token:", type="password")
        if 'hf_token' in st.session_state.keys():
            st.success('HF token already provided!', icon='✅')
            hf_token = st.session_state.hf_token
        else:
            hf_token = st.text_input('Enter your Hugging Face token:', type='password')
            if not (hf_token.startswith('hf_') and len(hf_token)==37):
                st.warning('please enter a valid token', icon='⚠️')
            else:
                st.success('Proceed to finetuning your model!', icon='👉')
                st.session_state.hf_token = hf_token
        
        # PEFT model name
        peft_model_name = st.text_input(
            "Enter the PEFT model name:", 
            help="The name of the PEFT model should start with the username associated with the provided HF token, followed by '/'ß. E.g. 'username/peft-base-uncased'"
        )
        
        # Dataset selection
        dataset_option = st.radio("Choose dataset source:", ["Upload JSON", "Hugging Face Dataset"])
        
        if dataset_option == "Upload JSON":
            uploaded_file = st.file_uploader("Upload JSON dataset", type="json")
            if uploaded_file is not None:
                dataset = json.load(uploaded_file)
                st.success("Dataset uploaded successfully!")
        else:
            dataset_name = st.text_input(
                "Enter Hugging Face dataset name:",
                help="The dataset name should follow the format 'username/dataset-name'"
            )
            file_name = st.text_input(
                "Enter the specific file name:",
                help="The file name should include the exact JSON file to be used. E.g., xx.json"
            )
        
        # Finetuning parameters
        st.subheader("Finetuning parameters")
        lora_rank = st.number_input("LoRA rank", min_value=2, max_value=64, value=16, step=2)
        lora_alpha = st.number_input("LoRA alpha", min_value=2, max_value=64, value=16, step=2)
        target_modules = st.multiselect("Target modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"], default=["down_proj"])
        learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1e-3, value=1e-5, step=1e-6)
        optimizer_type = st.selectbox("Optimizer type", ["SGD", "Adam", "AdamW", "Adagrad", "Adadelta", "Adamax", "RMSprop"])
        momentum = st.number_input("Momentum", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        nesterov = st.checkbox("Nesterov")
        max_steps = st.number_input("Max steps", min_value=1000, max_value=100000, value=10000, step=1000)
        
        # Start finetuning button
        if st.button("Start Finetuning"):
            if not hf_token:
                st.error("Please enter your Hugging Face token.")
            elif dataset_option == "Upload JSON" and uploaded_file is None:
                st.error("Please upload a JSON dataset.")
            elif dataset_option == "Hugging Face Dataset" and (not dataset_name or not file_name):
                st.error("Please enter a Hugging Face dataset name and file name.")
            else:
                # Prepare the request data
                request_data = {
                    "token": hf_token,
                    "peft_model_id": peft_model_name,
                    "dataset_option": dataset_option,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "target_modules": target_modules,
                    "learning_rate": learning_rate,
                    "optimizer_type": optimizer_type,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "nesterov": nesterov,
                    "max_steps": max_steps,
                }
                
                if dataset_option == "Upload JSON":
                    request_data["dataset"] = dataset
                else:
                    request_data["dataset_name"] = dataset_name
                    request_data["file_name"] = file_name
                
                print("---Front: here is request data----")
                print(request_data)
                # Send finetuning request to FastAPI server
                with st.spinner("Finetuning in progress..."):
                    response = requests.post(FINETUNE_URL, json=request_data)

                result = response.json()
                if response.status_code == 200:
                    st.success("Finetuning completed successfully!")
                else:
                    error_msg = result.get("detail", "Unknown error occurred.")
                    st.error(f"Finetuning failed. {error_msg}")

if page == "Chat":
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Running..."):
                response = generate_llama3_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
elif page == "Finetune":
    st.write("Use the sidebar to configure and start finetuning.")