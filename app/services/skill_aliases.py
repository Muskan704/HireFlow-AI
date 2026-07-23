"""
Skill alias normalization.

Purpose
-------
The hard filter (Stage 3) does strict lowercase string matching between JD
must-have skills and resume skills. That's correct for determinism, but it
produces false rejections when a candidate/resume uses a different surface
form of the same skill than the JD does.

Examples of what this solves:
  Resume says "K8s"         — JD says "Kubernetes"           → same skill
  Resume says "ReactJS"     — JD says "React"                → same skill
  Resume says "Postgres"    — JD says "PostgreSQL"           → same skill
  Resume says "ML"          — JD says "Machine Learning"     → same skill
  Resume says "Gen AI"      — JD says "Generative AI"        → same skill
  Resume says "LangChain"   — JD says "Langchain"            → same skill

Design rules
------------
1. Every mapping is hand-curated. No fuzzy/semantic matching here.
   Fuzzy matching belongs in Stage 4 (embeddings). Stage 3 is deterministic.
2. Canonical form = the string recruiters write in JDs most commonly.
   Reject reasons shown to recruiters use these canonical strings.
3. Add to this file when you observe false rejections in production logs.
   Do NOT auto-generate with an LLM. Auditable and deterministic.
4. When in doubt about whether two things are the same skill: keep them
   separate. It's better to occasionally fail a match than to incorrectly
   pass one. Use Stage 4 semantic similarity for borderline cases.

Usage
-----
    from app.services.skill_aliases import normalize_skill, normalize_skill_list

    resume.normalised_skills = normalize_skill_list(resume.skills)
    jd.normalised_must_have = normalize_skill_list(jd.must_have_skills)

Integration point
-----------------
Drop this into app/services/skill_aliases.py, then in schemas.py replace the
plain `.lower()` computed-field logic for normalised_skills /
normalised_must_have_skills with a call to normalize_skill_list(). No other
code changes needed — hard_filter.py already compares normalised fields.
"""

from __future__ import annotations
import re

# ─────────────────────────────────────────────────────────────────────────────
# ALIAS TABLE
# Format: "canonical form" -> {all surface forms that mean the same thing}
# All entries are matched case-insensitively. Canonical form is always in the
# set too. Canonical form is what appears in reject reasons shown to
# recruiters, so pick what recruiters write in JDs, not the full formal name.
# ─────────────────────────────────────────────────────────────────────────────

_SKILL_ALIAS_GROUPS: dict[str, set[str]] = {

    # ── CLOUD PLATFORMS ───────────────────────────────────────────────────────
    "aws": {
        "aws", "amazon web services", "amazon aws",
        "aws cloud", "amazon cloud",
    },
    "gcp": {
        "gcp", "google cloud platform", "google cloud",
        "google cloud services", "gc", "gcloud",
    },
    "azure": {
        "azure", "microsoft azure", "azure cloud",
        "ms azure", "azure services",
    },
    "ibm cloud": {
        "ibm cloud", "ibm bluemix", "bluemix",
    },
    "oracle cloud": {
        "oracle cloud", "oci", "oracle cloud infrastructure",
    },
    "alibaba cloud": {
        "alibaba cloud", "aliyun",
    },

    # ── CLOUD SERVICES (AWS-specific) ─────────────────────────────────────────
    "aws lambda": {
        "aws lambda", "lambda", "serverless lambda",
    },
    "aws ec2": {
        "aws ec2", "ec2", "elastic compute cloud",
    },
    "aws s3": {
        "aws s3", "s3", "amazon s3", "simple storage service",
    },
    "aws rds": {
        "aws rds", "rds", "relational database service",
    },
    "aws dynamodb": {
        "aws dynamodb", "dynamodb",
    },
    "aws sqs": {
        "aws sqs", "sqs", "simple queue service",
    },
    "aws sns": {
        "aws sns", "sns", "simple notification service",
    },
    "aws cloudformation": {
        "aws cloudformation", "cloudformation",
    },
    "aws eks": {
        "aws eks", "eks", "elastic kubernetes service",
    },
    "aws ecs": {
        "aws ecs", "ecs", "elastic container service",
    },
    "aws glue": {
        "aws glue", "glue",
    },
    "aws redshift": {
        "aws redshift", "redshift",
    },
    "aws kinesis": {
        "aws kinesis", "kinesis",
    },
    "aws sagemaker": {
        "aws sagemaker", "sagemaker",
    },
    "aws bedrock": {
        "aws bedrock", "bedrock",
    },
    "aws cognito": {
        "aws cognito", "cognito",
    },
    "aws api gateway": {
        "aws api gateway", "api gateway",
    },

    # ── CLOUD SERVICES (GCP-specific) ─────────────────────────────────────────
    "bigquery": {
        "bigquery", "bq", "google bigquery",
    },
    "google kubernetes engine": {
        "google kubernetes engine", "gke",
    },
    "google cloud run": {
        "google cloud run", "cloud run",
    },
    "google cloud functions": {
        "google cloud functions", "cloud functions",
    },
    "vertex ai": {
        "vertex ai", "google vertex ai",
    },

    # ── CLOUD SERVICES (Azure-specific) ───────────────────────────────────────
    "azure devops": {
        "azure devops", "ado", "azure devops services",
    },
    "azure kubernetes service": {
        "azure kubernetes service", "aks",
    },
    "azure functions": {
        "azure functions",
    },
    "azure ml": {
        "azure ml", "azure machine learning", "azure mlops",
    },
    "azure blob storage": {
        "azure blob storage", "blob storage",
    },
    "azure cosmos db": {
        "azure cosmos db", "cosmos db",
    },
    "azure sql": {
        "azure sql", "azure sql database",
    },
    "azure openai": {
        "azure openai", "azure openai service",
    },

    # ── CONTAINER & ORCHESTRATION ─────────────────────────────────────────────
    "kubernetes": {
        "kubernetes", "k8s", "k 8 s",
    },
    "docker": {
        "docker", "docker container", "containerization", "containerisation",
        "docker engine",
    },
    "docker compose": {
        "docker compose", "docker-compose",
    },
    "helm": {
        "helm", "helm charts",
    },
    "openshift": {
        "openshift", "red hat openshift", "ocp",
    },
    "argo cd": {
        "argo cd", "argocd", "argo",
    },
    "flux": {
        "flux", "fluxcd",
    },

    # ── CI/CD & DEVOPS ────────────────────────────────────────────────────────
    "ci/cd": {
        "ci/cd", "cicd", "ci cd",
        "continuous integration/continuous deployment",
        "continuous integration", "continuous deployment",
        "continuous integration and continuous delivery",
        "continuous delivery",
    },
    "jenkins": {
        "jenkins", "jenkins ci",
    },
    "github actions": {
        "github actions", "gh actions",
    },
    "gitlab ci": {
        "gitlab ci", "gitlab ci/cd", "gitlab pipelines",
    },
    "circleci": {
        "circleci", "circle ci",
    },
    "travis ci": {
        "travis ci", "travisci",
    },
    "teamcity": {
        "teamcity", "team city",
    },
    "bamboo": {
        "bamboo", "atlassian bamboo",
    },
    "ansible": {
        "ansible", "ansible playbooks",
    },
    "terraform": {
        "terraform", "tf", "iac", "infrastructure as code",
        "infrastructure-as-code",
    },
    "pulumi": {
        "pulumi",
    },
    "chef": {
        "chef", "chef infra",
    },
    "puppet": {
        "puppet", "puppet enterprise",
    },
    "prometheus": {
        "prometheus", "prometheus monitoring",
    },
    "grafana": {
        "grafana", "grafana dashboards",
    },
    "datadog": {
        "datadog", "data dog",
    },
    "new relic": {
        "new relic", "newrelic",
    },
    "splunk": {
        "splunk",
    },
    "elk stack": {
        "elk stack", "elk", "elasticsearch logstash kibana",
        "elastic stack",
    },
    "elasticsearch": {
        "elasticsearch", "elastic search", "es",
    },
    "logstash": {
        "logstash",
    },
    "kibana": {
        "kibana",
    },

    # ── VERSION CONTROL ───────────────────────────────────────────────────────
    "git": {
        "git", "github", "git version control", "source control with git",
    },
    "gitlab": {
        "gitlab", "git lab",
    },
    "bitbucket": {
        "bitbucket", "bit bucket", "atlassian bitbucket",
    },
    "svn": {
        "svn", "subversion",
    },
    "mercurial": {
        "mercurial", "hg",
    },

    # ── PROGRAMMING LANGUAGES ─────────────────────────────────────────────────
    "python": {
        "python", "python3", "python 3", "py", "python2", "python 2",
    },
    "javascript": {
        "javascript", "js", "ecmascript", "es6", "es2015", "es2016",
        "es2017", "es2018", "es2019", "es2020", "es2021", "es2022",
        "vanilla js", "vanillajs",
    },
    "typescript": {
        "typescript", "ts",
    },
    "java": {
        "java", "java se", "java ee", "java 8", "java 11", "java 17",
        "java 21", "core java",
    },
    "kotlin": {
        "kotlin", "kotlin jvm",
    },
    "scala": {
        "scala",
    },
    "c": {
        "c", "c programming", "c language",
    },
    "c++": {
        "c++", "cpp", "c plus plus",
    },
    "c#": {
        "c#", "csharp", "c sharp", "c# .net",
    },
    "go": {
        "go", "golang",
    },
    "rust": {
        "rust", "rust lang", "rust programming",
    },
    "ruby": {
        "ruby", "ruby on rails", "ror",
    },
    "php": {
        "php", "php 8", "php 7",
    },
    "swift": {
        "swift", "swift ui", "swiftui",
    },
    "objective-c": {
        "objective-c", "objc", "objective c",
    },
    "r": {
        "r", "r language", "r programming", "r stats",
    },
    "matlab": {
        "matlab",
    },
    "dart": {
        "dart",
    },
    "perl": {
        "perl",
    },
    "haskell": {
        "haskell",
    },
    "elixir": {
        "elixir",
    },
    "erlang": {
        "erlang",
    },
    "clojure": {
        "clojure",
    },
    "julia": {
        "julia", "julia lang",
    },
    "groovy": {
        "groovy",
    },
    "shell scripting": {
        "shell scripting", "bash", "bash scripting", "shell script",
        "unix shell", "linux shell", "bash shell", "sh scripting",
        "zsh",
    },
    "powershell": {
        "powershell", "ps1", "windows powershell",
    },

    # ── WEB FRAMEWORKS & LIBRARIES ────────────────────────────────────────────
    "react": {
        "react", "reactjs", "react.js", "react js", "react native",
    },
    "react native": {
        "react native", "reactnative",
    },
    "next.js": {
        "next.js", "nextjs", "next js",
    },
    "vue.js": {
        "vue.js", "vuejs", "vue", "vue js",
    },
    "nuxt.js": {
        "nuxt.js", "nuxtjs", "nuxt",
    },
    "angular": {
        "angular", "angularjs", "angular.js", "angular 2",
        "angular 14", "angular 15", "angular 16", "angular 17",
    },
    "svelte": {
        "svelte", "sveltekit",
    },
    "ember.js": {
        "ember.js", "emberjs", "ember",
    },
    "backbone.js": {
        "backbone.js", "backbonejs",
    },
    "jquery": {
        "jquery", "jquery.js",
    },
    "redux": {
        "redux", "redux toolkit", "rtk",
    },
    "node.js": {
        "node.js", "nodejs", "node js", "node",
    },
    "express.js": {
        "express.js", "expressjs", "express", "express js",
    },
    "fastapi": {
        "fastapi", "fast api",
    },
    "flask": {
        "flask", "flask api",
    },
    "django": {
        "django", "django rest framework", "drf",
    },
    "spring": {
        "spring", "spring boot", "spring framework",
        "spring mvc", "spring cloud",
    },
    "spring boot": {
        "spring boot", "springboot",
    },
    "laravel": {
        "laravel",
    },
    "symfony": {
        "symfony",
    },
    "rails": {
        "rails", "ruby on rails", "ror",
    },
    "asp.net": {
        "asp.net", "aspnet", "asp net", "asp.net core", "dotnet",
        ".net", ".net core",
    },
    "fastify": {
        "fastify",
    },
    "nest.js": {
        "nest.js", "nestjs", "nest js",
    },
    "graphql": {
        "graphql", "graph ql",
    },

    # ── MOBILE DEVELOPMENT ────────────────────────────────────────────────────
    "flutter": {
        "flutter",
    },
    "android": {
        "android", "android development", "android sdk",
    },
    "ios": {
        "ios", "ios development", "ios sdk", "iphone development",
    },
    "xamarin": {
        "xamarin",
    },
    "ionic": {
        "ionic",
    },

    # ── AI / ML / DATA SCIENCE ────────────────────────────────────────────────
    "machine learning": {
        "machine learning", "ml",
    },
    "deep learning": {
        "deep learning", "dl",
    },
    "artificial intelligence": {
        "artificial intelligence", "ai",
    },
    "natural language processing": {
        "natural language processing", "nlp",
    },
    "computer vision": {
        "computer vision", "cv",
    },
    "generative ai": {
        "generative ai", "gen ai", "genai", "generative artificial intelligence",
    },
    "llm": {
        "llm", "llms", "large language model", "large language models",
        "large language model llm", "large language models llms",
        "large language models (llms)", "large language model (llm)",
    },
    "rag": {
        "rag", "retrieval augmented generation",
        "retrieval-augmented generation",
    },
    "prompt engineering": {
        "prompt engineering", "prompting", "llm prompting",
    },
    "fine-tuning": {
        "fine-tuning", "fine tuning", "finetuning", "llm fine-tuning",
        "model fine-tuning",
    },
    "tensorflow": {
        "tensorflow", "tf", "tensorflow 2", "tf2",
    },
    "pytorch": {
        "pytorch", "torch",
    },
    "keras": {
        "keras",
    },
    "scikit-learn": {
        "scikit-learn", "sklearn", "scikit learn",
    },
    "xgboost": {
        "xgboost", "xgb", "extreme gradient boosting",
    },
    "lightgbm": {
        "lightgbm", "lgbm", "light gbm",
    },
    "catboost": {
        "catboost",
    },
    "opencv": {
        "opencv", "cv2", "open cv",
    },
    "hugging face": {
        "hugging face", "huggingface", "hf",
    },
    "transformers": {
        "transformers", "hugging face transformers",
    },
    "langchain": {
        "langchain", "lang chain",
    },
    "llamaindex": {
        "llamaindex", "llama index", "llama-index",
    },
    "openai": {
        "openai", "open ai", "openai api", "chatgpt api",
    },
    "gpt": {
        "gpt", "gpt-3", "gpt-4", "gpt-4o", "gpt4", "gpt3",
    },
    "llama": {
        "llama", "llama 2", "llama 3", "llama2", "llama3", "meta llama",
    },
    "anthropic claude": {
        "anthropic claude", "claude", "claude api",
    },
    "mistral": {
        "mistral", "mistral ai", "mistral 7b",
    },
    "mlops": {
        "mlops", "ml ops", "ml operations",
    },
    "mlflow": {
        "mlflow", "ml flow",
    },
    "kubeflow": {
        "kubeflow", "kube flow",
    },
    "airflow": {
        "airflow", "apache airflow",
    },
    "prefect": {
        "prefect",
    },
    "dvc": {
        "dvc", "data version control",
    },
    "weights & biases": {
        "weights & biases", "wandb", "w&b", "weights and biases",
    },
    "feast": {
        "feast", "feature store",
    },
    "feature engineering": {
        "feature engineering", "feature extraction",
    },
    "time series analysis": {
        "time series analysis", "time series", "time-series",
    },
    "reinforcement learning": {
        "reinforcement learning", "rl",
    },
    "federated learning": {
        "federated learning",
    },
    "data science": {
        "data science", "ds",
    },
    "data analysis": {
        "data analysis", "data analytics",
    },
    "statistics": {
        "statistics", "statistical analysis", "stats",
    },

    # ── DATA ENGINEERING ──────────────────────────────────────────────────────
    "apache spark": {
        "apache spark", "spark", "pyspark", "spark sql",
    },
    "apache kafka": {
        "apache kafka", "kafka",
    },
    "apache flink": {
        "apache flink", "flink",
    },
    "apache hadoop": {
        "apache hadoop", "hadoop", "hdfs",
    },
    "apache hive": {
        "apache hive", "hive",
    },
    "apache beam": {
        "apache beam", "beam",
    },
    "dbt": {
        "dbt", "data build tool",
    },
    "etl": {
        "etl", "extract transform load", "elt",
    },
    "data pipelines": {
        "data pipelines", "data pipeline",
    },
    "databricks": {
        "databricks", "data bricks",
    },
    "snowflake": {
        "snowflake", "snowflake db", "snowflake data warehouse",
    },
    "redshift": {
        "redshift", "aws redshift", "amazon redshift",
    },
    "dask": {
        "dask",
    },
    "polars": {
        "polars",
    },
    "data warehouse": {
        "data warehouse", "dwh", "data warehousing",
    },
    "data lake": {
        "data lake", "datalake",
    },
    "data lakehouse": {
        "data lakehouse", "lakehouse",
    },
    "delta lake": {
        "delta lake", "delta",
    },
    "iceberg": {
        "iceberg", "apache iceberg",
    },

    # ── DATABASES ─────────────────────────────────────────────────────────────
    "sql": {
        "sql", "structured query language",
    },
    "postgresql": {
        "postgresql", "postgres", "psql", "pg",
    },
    "mysql": {
        "mysql", "my sql",
    },
    "microsoft sql server": {
        "microsoft sql server", "mssql", "ms sql", "sql server",
        "t-sql", "tsql",
    },
    "oracle database": {
        "oracle database", "oracle db", "oracle", "pl/sql",
    },
    "sqlite": {
        "sqlite", "sqlite3",
    },
    "mongodb": {
        "mongodb", "mongo", "mongo db",
    },
    "redis": {
        "redis",
    },
    "cassandra": {
        "cassandra", "apache cassandra",
    },
    "dynamodb": {
        "dynamodb", "aws dynamodb",
    },
    "couchdb": {
        "couchdb", "couch db",
    },
    "neo4j": {
        "neo4j", "neo 4j", "graph database",
    },
    "influxdb": {
        "influxdb", "influx db",
    },
    "clickhouse": {
        "clickhouse", "click house",
    },
    "cockroachdb": {
        "cockroachdb", "cockroach db",
    },
    "supabase": {
        "supabase",
    },
    "firebase": {
        "firebase", "google firebase",
    },
    "nosql": {
        "nosql", "no sql",
    },
    "vector database": {
        "vector database", "vector databases", "vector db", "vector dbs",
        "vectordb", "vectordbs",
    },
    "chroma": {
        "chroma", "chromadb", "chroma db",
    },
    "pinecone": {
        "pinecone",
    },
    "weaviate": {
        "weaviate",
    },
    "qdrant": {
        "qdrant",
    },
    "faiss": {
        "faiss", "facebook ai similarity search",
    },
    "milvus": {
        "milvus",
    },
    "pgvector": {
        "pgvector", "pg vector",
    },

    # ── MESSAGING / STREAMING ─────────────────────────────────────────────────
    "rabbitmq": {
        "rabbitmq", "rabbit mq",
    },
    "apache pulsar": {
        "apache pulsar", "pulsar",
    },
    "nats": {
        "nats", "nats messaging",
    },
    "celery": {
        "celery",
    },
    "mqtt": {
        "mqtt",
    },

    # ── API & PROTOCOLS ───────────────────────────────────────────────────────
    "rest api": {
        "rest api", "rest apis", "restful api", "restful apis",
        "rest", "restful", "representational state transfer",
        "rest/graphql api", "rest/graphql apis", "rest graphql api",
        "rest graphql apis",
    },
    "graphql": {
        "graphql", "graph ql",
    },
    "grpc": {
        "grpc", "g-rpc", "google rpc",
    },
    "websocket": {
        "websocket", "websockets", "web socket", "web sockets",
    },
    "openapi": {
        "openapi", "open api", "swagger",
    },
    "api design": {
        "api design", "api development",
    },
    "microservices": {
        "microservices", "micro services", "microservice architecture",
    },
    "service mesh": {
        "service mesh", "istio",
    },

    # ── SECURITY ──────────────────────────────────────────────────────────────
    "oauth": {
        "oauth", "oauth 2.0", "oauth2",
    },
    "jwt": {
        "jwt", "json web token", "json web tokens",
    },
    "ssl/tls": {
        "ssl/tls", "ssl", "tls", "https",
    },
    "penetration testing": {
        "penetration testing", "pen testing", "pentesting",
    },
    "cybersecurity": {
        "cybersecurity", "cyber security", "infosec", "information security",
    },
    "devsecops": {
        "devsecops", "dev sec ops",
    },
    "siem": {
        "siem", "security information and event management",
    },
    "iam": {
        "iam", "identity and access management", "identity management",
    },
    "zero trust": {
        "zero trust", "zero-trust",
    },

    # ── DATA VISUALIZATION ────────────────────────────────────────────────────
    "tableau": {
        "tableau",
    },
    "power bi": {
        "power bi", "powerbi", "microsoft power bi",
    },
    "looker": {
        "looker", "looker studio", "google looker",
    },
    "matplotlib": {
        "matplotlib", "mpl",
    },
    "seaborn": {
        "seaborn",
    },
    "plotly": {
        "plotly",
    },
    "d3.js": {
        "d3.js", "d3", "d3js", "data driven documents",
    },
    "qlik": {
        "qlik", "qlikview", "qlik sense",
    },
    "metabase": {
        "metabase",
    },

    # ── TESTING ───────────────────────────────────────────────────────────────
    "unit testing": {
        "unit testing", "unit tests",
    },
    "pytest": {
        "pytest", "py.test",
    },
    "junit": {
        "junit", "junit 5",
    },
    "jest": {
        "jest", "jest testing",
    },
    "selenium": {
        "selenium", "selenium webdriver",
    },
    "cypress": {
        "cypress", "cypress testing",
    },
    "playwright": {
        "playwright",
    },
    "mocha": {
        "mocha", "mocha.js",
    },
    "postman": {
        "postman",
    },
    "tdd": {
        "tdd", "test driven development", "test-driven development",
    },
    "bdd": {
        "bdd", "behaviour driven development", "behavior driven development",
    },

    # ── OFFICE & PRODUCTIVITY ─────────────────────────────────────────────────
    "excel": {
        "excel", "microsoft excel", "ms excel",
    },
    "word": {
        "word", "microsoft word", "ms word",
    },
    "powerpoint": {
        "powerpoint", "microsoft powerpoint", "ms powerpoint", "ppt",
    },
    "outlook": {
        "outlook", "microsoft outlook", "ms outlook",
    },
    "teams": {
        "teams", "microsoft teams", "ms teams",
    },
    "office 365": {
        "office 365", "microsoft 365", "m365", "o365", "microsoft office",
    },
    "google workspace": {
        "google workspace", "g suite", "gsuite", "google docs",
        "google sheets", "google drive",
    },
    "jira": {
        "jira", "atlassian jira",
    },
    "confluence": {
        "confluence", "atlassian confluence",
    },
    "slack": {
        "slack",
    },
    "notion": {
        "notion",
    },
    "trello": {
        "trello", "atlassian trello",
    },
    "asana": {
        "asana",
    },
    "monday.com": {
        "monday.com", "monday", "mondaycom",
    },

    # ── ARCHITECTURE & DESIGN PATTERNS ────────────────────────────────────────
    "system design": {
        "system design", "systems design",
    },
    "design patterns": {
        "design patterns",
    },
    "oop": {
        "oop", "object oriented programming", "object-oriented programming",
        "oops", "object oriented design",
    },
    "functional programming": {
        "functional programming", "fp",
    },
    "event-driven architecture": {
        "event-driven architecture", "eda", "event driven architecture",
        "event-driven",
    },
    "serverless": {
        "serverless", "serverless architecture", "faas",
        "function as a service",
    },
    "solid principles": {
        "solid principles", "solid",
    },
    "domain driven design": {
        "domain driven design", "ddd",
    },
    "cqrs": {
        "cqrs", "command query responsibility segregation",
    },
    "event sourcing": {
        "event sourcing",
    },
    "clean architecture": {
        "clean architecture",
    },
    "hexagonal architecture": {
        "hexagonal architecture", "ports and adapters",
    },

    # ── DATA FORMATS & PROTOCOLS ──────────────────────────────────────────────
    "json": {
        "json", "javascript object notation",
    },
    "xml": {
        "xml", "extensible markup language",
    },
    "yaml": {
        "yaml", "yml",
    },
    "protobuf": {
        "protobuf", "protocol buffers",
    },
    "avro": {
        "avro", "apache avro",
    },
    "parquet": {
        "parquet", "apache parquet",
    },
    "orc": {
        "orc", "apache orc",
    },

    # ── DATA SCIENCE TOOLS ────────────────────────────────────────────────────
    "pandas": {
        "pandas", "pandas dataframe",
    },
    "numpy": {
        "numpy", "np",
    },
    "scipy": {
        "scipy",
    },
    "jupyter": {
        "jupyter", "jupyter notebook", "jupyter lab", "jupyterlab",
        "ipython",
    },

    # ── FRONTEND TOOLS & CONCEPTS ─────────────────────────────────────────────
    "html": {
        "html", "html5",
    },
    "css": {
        "css", "css3",
    },
    "tailwind css": {
        "tailwind css", "tailwind", "tailwindcss",
    },
    "bootstrap": {
        "bootstrap", "bootstrap css",
    },
    "sass": {
        "sass", "scss",
    },
    "webpack": {
        "webpack",
    },
    "vite": {
        "vite",
    },
    "figma": {
        "figma",
    },
    "ui/ux": {
        "ui/ux", "ui ux", "uiux",
    },
    "responsive design": {
        "responsive design", "responsive web design",
    },

    # ── PROGRAMMING CONCEPTS ──────────────────────────────────────────────────
    "data structures": {
        "data structures", "data structures and algorithms", "dsa",
    },
    "algorithms": {
        "algorithms", "algo",
    },
    "concurrency": {
        "concurrency", "concurrent programming", "multithreading",
        "multi-threading", "parallelism",
    },
    "orm": {
        "orm", "object relational mapping", "object-relational mapping",
    },

    # ── AGILE & PROJECT MANAGEMENT ────────────────────────────────────────────
    "agile": {
        "agile", "agile methodology", "agile development",
    },
    "scrum": {
        "scrum", "scrum methodology",
    },
    "kanban": {
        "kanban",
    },
    "devops": {
        "devops", "dev ops",
    },
    "sre": {
        "sre", "site reliability engineering",
    },
    "product management": {
        "product management", "pm",
    },

    # ── BLOCKCHAIN / WEB3 ─────────────────────────────────────────────────────
    "blockchain": {
        "blockchain", "block chain",
    },
    "solidity": {
        "solidity",
    },
    "ethereum": {
        "ethereum", "eth",
    },
    "web3": {
        "web3", "web 3", "web3.js",
    },
    "smart contracts": {
        "smart contracts", "smart contract",
    },

    # ── EMBEDDED / IOT / LOW-LEVEL ────────────────────────────────────────────
    "embedded systems": {
        "embedded systems", "embedded c",
    },
    "iot": {
        "iot", "internet of things",
    },
    "fpga": {
        "fpga", "field programmable gate array",
    },
    "arduino": {
        "arduino",
    },
    "raspberry pi": {
        "raspberry pi", "raspi",
    },
    "rtos": {
        "rtos", "real time operating system", "real-time os",
    },

    # ── NETWORKING ────────────────────────────────────────────────────────────
    "tcp/ip": {
        "tcp/ip", "tcp ip", "transmission control protocol",
    },
    "dns": {
        "dns", "domain name system",
    },
    "load balancing": {
        "load balancing", "load balancer",
    },
    "nginx": {
        "nginx", "nginx webserver",
    },
    "apache": {
        "apache", "apache http server", "apache httpd",
    },
    "cdn": {
        "cdn", "content delivery network",
    },
    "vpn": {
        "vpn", "virtual private network",
    },

    # ── OPERATING SYSTEMS ─────────────────────────────────────────────────────
    "linux": {
        "linux", "gnu/linux", "linux os",
    },
    "unix": {
        "unix", "unix systems",
    },
    "ubuntu": {
        "ubuntu",
    },
    "centos": {
        "centos", "redhat", "red hat", "rhel",
    },
    "windows": {
        "windows", "windows server", "windows os",
    },
    "macos": {
        "macos", "mac os", "osx", "os x",
    },

    # ── CERTIFICATIONS (common abbreviations) ─────────────────────────────────
    "aws certified solutions architect": {
        "aws certified solutions architect", "aws solutions architect",
        "aws sa", "aws csa",
    },
    "aws certified developer": {
        "aws certified developer",
    },
    "google cloud professional": {
        "google cloud professional",
    },
    "cka": {
        "cka", "certified kubernetes administrator",
    },
    "ckad": {
        "ckad", "certified kubernetes application developer",
    },
    "pmp": {
        "pmp", "project management professional",
    },
    "cissp": {
        "cissp", "certified information systems security professional",
    },
    "ceh": {
        "ceh", "certified ethical hacker",
    },
    "azure administrator": {
        "azure administrator", "az-104", "azure admin",
    },

    # ── SOFT SKILLS (commonly appear in JDs and resumes) ──────────────────────
    "communication": {
        "communication", "communication skills", "verbal communication",
        "written communication",
    },
    "problem solving": {
        "problem solving", "problem-solving",
    },
    "critical thinking": {
        "critical thinking",
    },
    "team collaboration": {
        "team collaboration", "teamwork", "collaboration",
    },
    "leadership": {
        "leadership", "team leadership",
    },

    # ── COMMON ACRONYMS / ABBREVIATIONS ──────────────────────────────────────
    "saas": {
        "saas", "software as a service",
    },
    "paas": {
        "paas", "platform as a service",
    },
    "iaas": {
        "iaas", "infrastructure as a service",
    },
    "api": {
        "api", "application programming interface",
    },
    "sdk": {
        "sdk", "software development kit",
    },
    "ide": {
        "ide", "integrated development environment",
    },
    "crud": {
        "crud", "create read update delete",
    },
    "mvp": {
        "mvp", "minimum viable product",
    },
    "poc": {
        "poc", "proof of concept",
    },
    "sla": {
        "sla", "service level agreement",
    },
    "kpi": {
        "kpi", "key performance indicator",
    },
    "ui": {
        "ui", "user interface",
    },
    "ux": {
        "ux", "user experience",
    },
    "rpc": {
        "rpc", "remote procedure call",
    },
    "cli": {
        "cli", "command line interface", "command line",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# REVERSE LOOKUP: alias (lowercase) -> canonical form
# Built once at module load time. O(1) lookup during pipeline execution.
# ─────────────────────────────────────────────────────────────────────────────

_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias.lower(): canonical
    for canonical, aliases in _SKILL_ALIAS_GROUPS.items()
    for alias in aliases
}


# ─────────────────────────────────────────────────────────────────────────────
# PARENTHETICAL ABBREVIATION EXTRACTOR
# Handles the pattern: "Object-Oriented Programming (OOP)" -> also extracts "OOP"
# This covers arbitrary abbreviation patterns without requiring hand-coded entries
# for every possible full-name/abbreviation pair.
# ─────────────────────────────────────────────────────────────────────────────

_PAREN_ABBREVIATION_PATTERN = re.compile(r"\(([A-Za-z0-9&/+\-]{2,15})\)")


def _extract_parenthetical_abbreviations(skill: str) -> list[str]:
    """Return any short parenthetical abbreviations found in a skill string.

    Example: "Object-Oriented Programming (OOP)" -> ["OOP"]
    """
    return _PAREN_ABBREVIATION_PATTERN.findall(skill)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def normalize_skill(skill: str) -> str:
    """
    Normalize a single skill string to its canonical form.

    Falls back to a plain lowercase/trim if no alias mapping exists —
    this preserves current behavior for every skill not explicitly
    covered by the alias table.

    Examples:
        "K8s"              -> "kubernetes"
        "ReactJS"          -> "react"
        "Postgres"         -> "postgresql"
        "ML"               -> "machine learning"
        "gen ai"           -> "generative ai"
        "SomethingUnknown" -> "somethingunknown"   (passthrough)
    """
    cleaned = skill.strip().lower()
    return _ALIAS_TO_CANONICAL.get(cleaned, cleaned)


def normalize_skill_list(skills: list[str]) -> list[str]:
    """
    Normalize a list of skills, de-duplicating after normalization.

    For each skill, produces UP TO TWO normalized entries:
      1. The full string, normalized/alias-mapped.
      2. If the skill has a parenthetical abbreviation (e.g. "(OOP)"),
         that abbreviation, ALSO alias-mapped, as a separate entry.

    Example:
      "Object-Oriented Programming (OOP)"
        -> ["oop", "oop"]  (deduped to ["oop"])

      "Natural Language Processing (NLP)"
        -> ["natural language processing", "nlp"]
    """
    seen: list[str] = []
    seen_set: set[str] = set()

    for skill in skills:
        primary = normalize_skill(skill)
        if primary not in seen_set:
            seen.append(primary)
            seen_set.add(primary)

        for abbr in _extract_parenthetical_abbreviations(skill):
            normalized_abbr = normalize_skill(abbr)
            if normalized_abbr not in seen_set:
                seen.append(normalized_abbr)
                seen_set.add(normalized_abbr)

    return seen


def skills_match(resume_skill: str, jd_skill: str) -> bool:
    """
    Check if two skill strings refer to the same skill after normalization.
    Use this for one-off comparisons. For bulk matching, pre-normalize
    both lists with normalize_skill_list() and use set intersection.

    Examples:
        skills_match("K8s", "Kubernetes")   -> True
        skills_match("ReactJS", "React")    -> True
        skills_match("AWS", "GCP")          -> False
    """
    return normalize_skill(resume_skill) == normalize_skill(jd_skill)


def get_canonical_form(skill: str) -> str:
    """Return just the canonical form for a skill (alias of normalize_skill)."""
    return normalize_skill(skill)


def get_all_aliases(canonical_skill: str) -> set[str]:
    """
    Return all known surface forms for a given skill.
    Useful for debugging — shows everything we consider equivalent.

    Example:
        get_all_aliases("kubernetes") -> {"kubernetes", "k8s", "k 8 s"}
    """
    canonical_lower = canonical_skill.strip().lower()
    # Find via canonical key
    if canonical_lower in _SKILL_ALIAS_GROUPS:
        return _SKILL_ALIAS_GROUPS[canonical_lower]
    # Find via alias
    if canonical_lower in _ALIAS_TO_CANONICAL:
        true_canonical = _ALIAS_TO_CANONICAL[canonical_lower]
        return _SKILL_ALIAS_GROUPS.get(true_canonical, {canonical_lower})
    return {canonical_lower}
