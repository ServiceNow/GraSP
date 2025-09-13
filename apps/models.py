try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "GraSP UI requires the optional 'ui' dependencies. "
        "Install them with: pip install 'grasp[ui]'"
    )
import yaml
import os
import asyncio
from pathlib import Path
import httpx
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta, timezone
from utils import check_model_status
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from grasp.utils.utils import load_model_config

UTC = timezone.utc
YAML_FILE = Path("../grasp/config/models.yaml")
USER_TZ = UTC
st.set_page_config(page_title="GraSP UI", layout="wide")

if "model_statuses" not in st.session_state:
    st.session_state["model_statuses"] = {}

if "last_updated" not in st.session_state:
    st.session_state["last_updated"] = None

if "draft_models" not in st.session_state:
    st.session_state["draft_models"] = {}

if "delete_confirm" not in st.session_state:
    st.session_state["delete_confirm"] = {}

if "new_params" not in st.session_state:
    st.session_state["new_params"] = []

if "active_models" not in st.session_state:
    st.session_state["active_models"] = []


def load_models():
    if os.path.exists(YAML_FILE):
        with open(YAML_FILE, "r") as file:
            return yaml.safe_load(file) or {}
    return {}


def save_models(models):
    with open(YAML_FILE, "w") as file:
        yaml.dump(models, file, default_flow_style=False)


def get_time_delta(delta: relativedelta):
    """Get a human-readable time delta like "2 hours ago" or "just now"."""
    for unit in ("years", "months", "days", "hours", "minutes", "seconds"):
        value = getattr(delta, unit)
        if value:
            return f"{value} {unit if value > 1 else unit[:-1]} ago"
    return "just now"


@st.fragment(run_every="1min")
def display_time_since(timestamp):
    st.markdown(
        f"🕒 Last updated {get_time_delta(relativedelta(datetime.now(UTC), timestamp))}",
        help=timestamp.astimezone(USER_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
    )


models = load_model_config()


async def update_model_statuses(models):
    async with httpx.AsyncClient() as session:
        tasks = [
            check_model_status(session, name, model) for name, model in models.items()
        ]
        statuses = await asyncio.gather(*tasks)
    return {
        name: ("🟢 Active" if is_active else "🔴 Inactive")
        for name, is_active in statuses
    }


async def initialize_model_statuses():
    st.session_state["model_statuses"] = await update_model_statuses(models)
    st.session_state["last_updated"] = datetime.now(UTC)


# Run the initialization only if model statuses are empty
if "model_statuses" not in st.session_state or not st.session_state["model_statuses"]:
    asyncio.run(initialize_model_statuses())  # ✅ This runs immediately on app start


st.markdown("# Models 🤖")

if st.button("🔄 Refresh Status"):
    st.session_state["model_statuses"] = asyncio.run(update_model_statuses(models))
    st.session_state["last_updated"] = datetime.now(UTC)
    st.rerun()

if st.session_state["last_updated"]:
    display_time_since(st.session_state["last_updated"])

for model_name in list(models.keys()):
    status = st.session_state["model_statuses"].get(model_name, "⚪ Checking...")
    with st.expander(
        f"**{model_name}** ({models[model_name].get('model_type', 'Unknown Type')}) {status}"
    ):
        st.json(models[model_name])

        # Delete Confirmation
        if model_name not in st.session_state["delete_confirm"]:
            st.session_state["delete_confirm"][model_name] = False

        if not st.session_state["delete_confirm"][model_name]:
            if st.button(f"🗑 Delete {model_name}", key=f"delete_{model_name}"):
                st.session_state["delete_confirm"][model_name] = True
                st.rerun()
        else:
            st.warning(
                f"Are you sure you want to delete **{model_name}**? This action cannot be undone."
            )
            if st.button(
                f"✅ Confirm Delete {model_name}", key=f"confirm_{model_name}"
            ):
                del models[model_name]
                save_models(models)
                del st.session_state["delete_confirm"][model_name]
                st.success(f"Model '{model_name}' deleted successfully!")
                st.rerun()

            if st.button("❌ Cancel", key=f"cancel_delete_{model_name}"):
                st.session_state["delete_confirm"][model_name] = False
                st.rerun()


if st.button("➕ Add Model"):
    st.session_state["show_add_model_form"] = True
    st.session_state["new_params"] = []  # Reset new parameters when adding a new model

if st.session_state.get("show_add_model_form", False):
    st.markdown("## Add New Model")

    with st.form("add_model_form"):
        new_model_name = st.text_input("Name").strip()
        new_model_type = st.selectbox(
            "Model Type", ["azure_openai", "tgi", "vllm", "mistralai"], index=0
        )
        new_model_model = st.text_input("Model")
        new_model_url = st.text_input("URL")
        new_model_api_key = st.text_input("API Key / Auth Token", type="password")
        new_model_api_version = st.text_input("API Version")

        if st.form_submit_button("➕ Add New Parameter"):
            st.session_state["new_params"].append({"key": "", "value": ""})
            st.rerun()

        to_delete1 = []
        for i, param in enumerate(st.session_state["new_params"]):
            colu1, colu2, colu3 = st.columns([3, 3, 1])
            with colu1:
                param["key"] = st.text_input(
                    f"Param {i + 1}", param["key"], key=f"key_{i}"
                )
            with colu2:
                param["value"] = st.text_input(
                    f"Value {i + 1}", param["value"], key=f"value_{i}"
                )
            with colu3:
                if st.form_submit_button(f"❌ Remove {i + 1}"):
                    to_delete1.append(i)

        for i in reversed(to_delete1):
            del st.session_state["new_params"][i]
            st.rerun()

        st.markdown("### Parameters")
        max_tokens = st.number_input("Max Tokens", min_value=1, value=500)
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, step=0.1)
        stop_tokens = st.text_area("Stop Tokens (comma-separated)").split(",")

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Save as Draft")
        with col2:
            cancel = st.form_submit_button("Cancel")

        if cancel:
            st.session_state["show_add_model_form"] = False
            st.rerun()

        if submitted:
            if not new_model_name:
                st.error("Model name cannot be empty!")
            elif (
                new_model_name in models
                or new_model_name in st.session_state["draft_models"]
            ):
                st.error("A model with this name already exists!")
            else:
                new_params = {
                    param["key"]: param["value"]
                    for param in st.session_state["new_params"]
                    if param["key"]
                }

                # Save as a draft
                st.session_state["draft_models"][new_model_name] = {
                    "model_type": new_model_type,
                    "model": new_model_model,
                    "url": new_model_url,
                    "api_key": new_model_api_key,
                    "api_version": new_model_api_version,
                    **new_params,
                    "parameters": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stop": stop_tokens,
                    },
                }
                st.success(f"Model '{new_model_name}' saved as draft!")
                st.session_state["show_add_model_form"] = False
                st.session_state["new_params"] = []
                st.rerun()

if st.session_state["draft_models"]:
    st.markdown("## Draft Models")
    for draft_name, draft_model in list(st.session_state["draft_models"].items()):
        with st.expander(
            f"**{draft_name}** ({draft_model.get('model_type', 'Unknown Type')}) - DRAFT"
        ):
            st.json(draft_model)

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Publish {draft_name}", key=f"publish_{draft_name}"):
                    models[draft_name] = draft_model
                    save_models(models)
                    del st.session_state["draft_models"][draft_name]
                    st.success(f"Model '{draft_name}' published!")
                    st.rerun()
            with col2:
                if st.button(
                    f"❌ Delete Draft {draft_name}", key=f"delete_draft_{draft_name}"
                ):
                    del st.session_state["draft_models"][draft_name]
                    st.rerun()
