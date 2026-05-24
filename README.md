# autoflow_ml
# AutoFlow ML: Autonomous Agentic Machine Learning Engine

AutoFlow ML is a fully autonomous, self-healing, multi-agent machine learning platform. It transforms raw datasets and natural language objectives into optimized, production-ready machine learning pipelines. 

By utilizing an advanced, multi-agent cooperative loop, the system automatically profiles data, drafts training strategies, writes execution code, self-heals runtime compilation or package errors, evaluates framework-agnostic metrics, and iteratively refines hyperparameters to achieve target optimization contracts.

---

## 🧠 System Architecture & Data Flow

AutoFlow ML operates as an event-driven optimization loop. The workflow shifts dynamically from static data profiling to an iterative, generative research phase:

```mermaid
graph TD
    A[Raw Dataset + User Request] --> B[Dataset Profiler / README Creator]
    SubGraph1[Data Ingestion Phase]
        B --> B1[(dataset_metadata.json)]
        B --> B2[Runtime Generated Dataset README]
    End

    B2 --> C[Planner Agent]
    C --> D[Initial Training Plan & Contract]

    subgraph Research Loop [Iterative Research & Optimization Loop]
        D --> E[Coding Agent]
        E --> F[generated_pipeline.py]
        F --> G((Isolated Execution Subprocess))
        
        G -- "Runtime Crash / Exception" --> H[Self-Healing Debugging Loop]
        H -- "Auto-Install / pip" --> G
        H -- "Analyze STDERR Traceback" --> I[Debugging Agent]
        I -- "Rewrite Fixed Code" --> F
        
        G -- "Execution Success" --> J[Dynamic Metric Parser]
        J --> K{Contract Target Achieved?}
    end

    K -- Yes --> L((End: Export Best Model & Logs))
    K -- No --> M[Performance Analyzer Agent]
    
    M --> N[Training Improvement Agent]
    N -- "Mutates Training Plan JSON" --> D
