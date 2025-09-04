try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "GraSP UI requires the optional 'ui' dependencies. "
        "Install them with: pip install 'grasp[ui]'"
    )

# Define the pages
Models = st.Page("models.py", title="Models", icon="🤖")
Tasks = st.Page("tasks.py", title="Tasks", icon="🗒️")
CreateTask = st.Page("create_new_task.py", title="Create new task", icon="➕")
CreateTaskFromTemplate = st.Page(
    "create_from_template.py", title="Create new task from template", icon="➕"
)

# Set up navigation
pg = st.navigation([Models, Tasks, CreateTask, CreateTaskFromTemplate])


# Run the selected page
pg.run()
