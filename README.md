# M3P1-Sistema-Multiagente
Sistema multiagente con un Agente Orquestador que clasifica la intención de la consulta del usuario. Esta clasificación activa un enrutamiento condicional que delega la tarea de recuperación al Agente RAG especializado correcto, para generar una respuesta contextualmente fundamentada.

## Variables de entorno

Variables de entorno esperadas (usar un archivo .env):

+ OPENAI_API_KEY=sk-...
+ LANGFUSE_PUBLIC_KEY=pk-lf-...
+ LANGFUSE_SECRET_KEY=sk-lf-...
+ LANGFUSE_HOST=https://cloud.langfuse.com o tu instancia self-hosted