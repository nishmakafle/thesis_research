:::mermaid
graph TD
    subgraph Input
        A[Input Image 224x224x3] --> B[ELA Preprocessing]
        B --> C[Gaussian Blur]
    end

    subgraph MobileNetV2 Base
        C --> D[MobileNetV2 Base Model]
        D --> E[Feature Maps]
    end

    subgraph Classification Head
        E --> F[Global Average Pooling 2D]
        F --> G[Dense Layer - 1024 units]
        G --> H[Dropout 0.5]
        H --> I[Dense Layer - 512 units]
        I --> J[Dropout 0.3]
        J --> K[Dense Layer - 1 unit]
        K --> L[Sigmoid Activation]
    end

    subgraph Output
        L --> M[Binary Classification]
        M --> N[Authentic]
        M --> O[Tampered]
    end

    style A fill:#f9f,stroke:#333
    style M fill:#9ff,stroke:#333
    style N fill:#9f9,stroke:#333
    style O fill:#f99,stroke:#333

    classDef baseModel fill:#e6f3ff,stroke:#333
    classDef classificationHead fill:#fff2e6,stroke:#333
    class D,E baseModel
    class F,G,H,I,J,K,L classificationHead
