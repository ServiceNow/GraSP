try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "GraSP UI requires the optional 'ui' dependencies. "
        "Install them with: pip install 'grasp[ui]'"
    )
import httpx
from openai import AsyncAzureOpenAI
from mistralai_azure import MistralAzure
from mistralai_azure.utils.retries import RetryConfig, BackoffStrategy
import aiohttp
import json


async def check_openai_status(session, model_name, model_data):
    try:
        client = AsyncAzureOpenAI(
            azure_endpoint=model_data["url"],
            api_key=model_data["api_key"],
            api_version=model_data["api_version"],
            timeout=model_data.get("timeout", 10),
            default_headers={"Connection": "close"},
        )

        # Sending test request
        completion = await client.chat.completions.create(
            model=model_data["model"],
            messages=[{"role": "system", "content": "Hello!"}],
            max_tokens=5,
            temperature=0.1,
        )

        # If no exception, model is active
        st.session_state["active_models"].append(model_name)
        return model_name, True

    except Exception:
        return model_name, False


async def check_mistralai_status(session, model_name, model_data):
    try:
        async_httpx_client = httpx.AsyncClient(
            http1=True, verify=True, timeout=model_data.get("timeout", 10)
        )

        retry_config = RetryConfig(
            strategy="backoff",
            retry_connection_errors=True,
            backoff=BackoffStrategy(
                initial_interval=1000,
                max_interval=1000,
                exponent=1.5,
                max_elapsed_time=10,
            ),
        )

        client = MistralAzure(
            azure_api_key=model_data["auth_token"],
            azure_endpoint=model_data["url"],
            async_client=async_httpx_client,
            retry_config=retry_config,
        )

        chat_response = await client.chat.complete_async(
            model=model_data["model"],
            messages=[{"role": "system", "content": "Hello!"}],
            max_tokens=5,
        )

        if chat_response and chat_response.choices:
            st.session_state["active_models"].append(model_name)
            return model_name, True

    except Exception as e:
        return model_name, False


async def check_tgi_status(session, model_name, model_data):
    try:
        payload = json.dumps({"inputs": "Hello!", "parameters": {"max_new_tokens": 5}})

        # Headers with authentication
        headers = {
            "Authorization": f"Bearer {model_data.get('auth_token', '')}",
            "Content-Type": "application/json",
        }

        timeout = model_data.get("timeout", 10)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                model_data["url"],
                data=payload,
                headers=headers,
                timeout=timeout,
                ssl=False,
            ) as response:
                if response.status == 200:
                    st.session_state["active_models"].append(model_name)
                    return model_name, True
                else:
                    error_text = await response.text()
                    return model_name, False

    except Exception as e:
        return model_name, False


async def check_vllm_status(session, model_name, model_data):
    try:
        url = model_data["url"]
        auth_token = model_data.get("auth_token", "").replace("Bearer ", "")
        model_serving_name = model_data.get("model_serving_name", model_name)

        async with httpx.AsyncClient(
            http1=True, verify=True, timeout=model_data.get("timeout", 10)
        ) as client:
            headers = {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
                "Connection": "close",
            }
            payload = {
                "model": model_serving_name,
                "prompt": "Hello!",
                "max_tokens": 5,
                "temperature": 0.1,
            }

            response = await client.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                st.session_state["active_models"].append(model_name)
                return model_name, True
            else:
                return model_name, False

    except Exception as e:
        return model_name, False


async def check_model_status(session, model_name, model_data):
    model_type = model_data.get("model_type", "unknown")
    if model_type == "azure_openai":
        return await check_openai_status(session, model_name, model_data)
    elif model_type == "mistralai":
        return await check_mistralai_status(session, model_name, model_data)
    elif model_type == "tgi":
        return await check_tgi_status(session, model_name, model_data)
    elif model_type == "vllm":
        return await check_vllm_status(session, model_name, model_data)
    return model_name, False
