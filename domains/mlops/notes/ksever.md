
## MLops with K8s
**沒有一個「單一」的 YAML 或 Helm Chart 可以做到這一切。**

相反地，您會使用 **Helm** 來\*\*「安裝平台」**（例如安裝 KServe, Kubeflow, Prometheus），然後您會使用**「YAMLs」**來**「執行您的工作流程」\*\*（例如提交一個訓練任務或部署一個模型）。

讓我們來建構一個這樣的範例。

**我們的 MLOps 技術棧 (Stack)：**

  * **KServe：** 用於模型推論 (Prediction Endpoints)。**它天生就是為了高流量和自動擴展而設計的。**
  * **Kubeflow Training Operator：** 用於訓練和測試 (Training, Testing)。
  * **Prometheus & Grafana：** 用於監控 (Online/Offline Metrics)。

-----

### 步驟一 (基礎設施)：使用 Helm 安裝 MLOps 平台

這一步是您（作為 MLOps 工程師）的「一次性」設定工作。

#### 0. 安裝 nvidia-gpu-operator
- 目的： 讓 K8s 節點能被辨識出 GPU 資源。
```
helm repo add nvidiachart https://nvidia.github.io/gpu-operator
helm install nvidia-gpu-operator nvidiachart/gpu-operator --namespace gpu-operator --create-namespace
```

#### 1\. 安裝 KServe (為了高效能推論)

KServe 會自動處理「模型部署」、「服務路由」、**「基於流量的自動擴展 (HPA/KPA)」以及「Online Metrics」**。

```bash
# 1. 新增 KServe 的 Helm 商店
helm repo add kserve https://kserve.github.io/charts

# 2. 安裝 KServe Operator (它會幫我們管理 InferenceService)
# 我們也順便安裝它推薦的 Knative (KPA)，這是實現高流量和「縮減至零」的關鍵
helm install kserve-operator kserve/kserve-operator --namespace kserve --create-namespace

# (KServe 依賴 Knative, Istio 等，為簡化，我們假設您已安裝)
```

#### 2\. 安裝 Kubeflow Training Operator (為了訓練)

這會讓您的 K8s 集群「看得懂」`TFJob`, `PytorchJob` 這種訓練專用的 YAML。

```bash
# 1. 新增 Kubeflow 的 Helm 商店
helm repo add kubeflow https://kubeflow.github.io/charts

# 2. 安裝 Training Operator
helm install training-operator kubeflow/training-operator --namespace kubeflow --create-namespace
```

#### 3\. 安裝 Prometheus & Grafana (為了監控)

我們使用 `kube-prometheus-stack`，這是一個包含所有元件的超大型 Chart。

```bash
# 1. 新增 Prometheus 的商店
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# 2. 安裝
helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace
```

**至此，您的 K8s 平台「升級」完畢。** 您的集群現在「看得懂」MLOps 專用的 YAML 了。

-----

### 步驟二 (工作流程)：使用 YAML 執行 MLOps 任務

現在，資料科學家（或您）可以開始「使用」這個平台了。

#### A. 訓練 (Training) & 測試 (Testing)

您**不再**需要建立 `Pod`。您會建立一個 `PytorchJob` (或 `TFJob`)。

**`train-model.yaml`**

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob  # <-- Helm 安裝的 Kubeflow Operator 讓 K8s 認識這個
metadata:
  name: pytorch-mnist-train
spec:
  pytorchReplicaSpecs:
    Worker:
      replicas: 1 # 1 個 worker
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: docker.io/kubeflow/pytorch-mnist:v1.0
              args: ["--save-model"] # 腳本參數：告訴它要保存模型
              # --- 連接到我們昨天的對話 ---
              resources:
                limits:
                  nvidia.com/gpu: 1 # <-- 請求 1 張 GPU
```

  * **如何執行：** `kubectl apply -f train-model.yaml`
  * **Offline Metrics & Testing：** 這個 `image` 內部的 Python 腳本會負責：
    1.  下載數據
    2.  `model.fit()` (Training)
    3.  `model.evaluate()` (Testing)
    4.  將測試結果 (例如 `val_accuracy: 0.98`) **print** 到 `stdout` (您可以用 `kubectl logs` 查看)。
    5.  將訓練好的模型 (例如 `model.pth`) **上傳到 S3 儲存桶**。

#### B. 推論 (Prediction) - 應付高流量

這是最精彩的部分。您**不需要**手動建立 `Deployment` + `Service` + `HPA`。

您只需要建立**一個** `InferenceService` 物件，KServe 會幫您搞定**一切**。

**`deploy-model.yaml`**

```yaml
apiVersion: "serving.kserve.io/v1beta1" # <-- Helm 安裝的 KServe Operator 讓 K8s 認識這個
kind: InferenceService
metadata:
  name: my-model-endpoint
spec:
  # KServe 會自動建立一個公開的 URL (Endpoint)
  predictor:
    # --- 應付高流量的關鍵 ---
    # 使用 KServe/Knative 的自動擴展器 (KPA)
    # 這裡的 "10" 表示：每個 Pod 同時處理 10 個請求就擴展
    annotations:
      autoscaling.kserve.io/metric: "concurrency"
      autoscaling.kserve.io/target: "10"
      
    # KServe 甚至支援 "Scale-to-Zero" (縮減至零)
    # 如果 10 分鐘內沒流量，自動把 Pod 縮減到 0，節省 GPU 成本
    # minReplicas: 0  <-- (預設)
    
    # KServe 自動處理 GPU (如果需要)
    # resources:
    #   limits:
    #     nvidia.com/gpu: 1

    # KServe 知道如何加載 S3 上的模型
    pytorch:
      storageUri: "s3://my-models/pytorch-mnist-train/1" # <-- 指向您上一步訓練好的模型
      # KServe 會自動下載模型，並用 TorchServe 啟動一個 API 伺服器
```

  * **如何執行：** `kubectl apply -f deploy-model.yaml`
  * **結果：**
    1.  KServe 會立刻去 S3 下載模型，啟動 1 個 Pod。
    2.  它會建立一個**公開的 Prediction Endpoint** (URL)。
    3.  **Online Metrics：** KServe **自動**在 `/metrics` 接口上暴露 **Prometheus** 格式的指標（例如：`request_count`, `request_latency`, `error_rate`）。
    4.  **高流量處理：** 當您的流量湧入，KPA (Knative Pod Autoscaler) 會偵測到每個 Pod 的併發請求超過了 10 個，它會**自動啟動**第 2, 3, 4... 個 Pod (直到 `maxReplicas`)。當流量下降，它會自動縮減 Pod 數量。

#### C. 監控 (Online & Offline Metrics)

  * **Helm 安裝的 Prometheus** 已經在自動抓取（scrape）K8s 的所有 `Service`。
  * **Online Metrics：**
      * Prometheus 會自動抓取 KServe 暴露的 `my-model-endpoint` 的 `/metrics` 接口。
      * 您只需要打開 **Helm 安裝的 Grafana**，新增一個儀表板，您就可以立刻拉出「P95 延遲」、「QPS (每秒請求數)」和「錯誤率」的即時圖表。
  * **Offline Metrics：**
      * Prometheus 也會抓取 `Kubeflow Training Operator` 的指標。
      * 您可以在 Grafana 儀表板上看到「歷史上所有 `PyTorchJob` 的成功/失敗次數」、「訓練 GPU 使用時長」等。

### 總結

| 層級  | 技術                         | 主要任務                         | Helm/YAML 關鍵命令                       |
| --- | -------------------------- | ---------------------------- | ------------------------------------ |
| 訓練層 | Kubeflow Training Operator | 建立 `PyTorchJob`，訓練模型上傳 S3    | `helm install training-operator`     |
| 推論層 | KServe + Knative           | 部署 `InferenceService`，自動擴展流量 | `helm install kserve-operator`       |
| 監控層 | Prometheus & Grafana       | 收集線上/離線 Metrics              | `helm install kube-prometheus-stack` |


1.  **Helm (安裝)：**
      * `helm install kserve` (用於推論 + 高流量)
      * `helm install training-operator` (用於訓練)
      * `helm install kube-prometheus-stack` (用於監控)
2.  **YAML (執行)：**
      * `kubectl apply -f train-job.yaml` (建立一個 `PyTorchJob`，請求 GPU，訓練完模型上傳 S3)。
      * `kubectl apply -f deploy-model.yaml` (建立一個 `InferenceService`，從 S3 下載模型，並**自動處理高流量擴展和 Online Metrics**)。