---
title: 'Fundamental Deep Learning 3D Reconstruction Techniques'
date: 2024-11-24
permalink: /3d-reconstruction/
tags:
  - 3D Reconstruction
  - Latent Diffusion Models
  - 3D Gaussian Splatting
  - NeRF
  - Side Project
---

**Authors:** Yonge Bai, Wong Lik Hang Kenny, Tsz Yin Twan

This article presents fundamental deep learning techniques for 3D reconstruction with the aim of sharing knowledge, I do not claim any credit for the techniques introduced next. If you are interested in using any of the discussed methods, please refer to the original works listed in the references section for more detailed information. 

**If you are looking for English version, please navigate to [https://arxiv.org/abs/2407.08137](https://arxiv.org/abs/2407.08137).**

這是一篇介紹Latent Diffusion Models, 3D Gaussian Splatting, NeRF如何用於三維重建的文章（使用了GPT等工具潤色）。Arxiv 版本已放在了 [https://arxiv.org/abs/2407.08137](https://arxiv.org/abs/2407.08137), 有興趣可以去查閱看看。如有疑問可以在 github 建立 issue 查詢。

# Introduction

三維重建（3D Reconstruction）是從影像和/或視訊資料中建立體積表面的過程。近年來，該領域在學術界和工業界都引起了極大的關注，廣泛應用於虛擬實境（VR）、擴增實境（AR）、自動駕駛和機器人等眾多領域。深度學習（Deep Learning，DL）已成為推動三維重建技術發展的核心力量，展現了在增強真實感和精確度方面的卓越成果。

本Article將深入探討基於深度學習的三維重建技術，重點在於神經輻射場（Neural Radiance Fields，NeRF）、潛在擴散模型（Latent Diffusion Models，LDM）和三維高斯散射（3D Gaussian Splatting）。我們將解剖這些技術背後的演算法，分析其優劣勢，並展望這一快速發展的領域的未來研究方向。

## Neural Radiance Field（NeRF）

神經輻射場（Neural Radiance Field，NeRF）是一種利用一組輸入視角對複雜場景進行新視角合成的方法，透過優化模型來近似連續的體積場景或表面。此方法使用多層感知機（MLP）表示體積，其輸入為5D向量 $$(x, y, z, \theta, \phi)$$，其中$$(x, y, z)$$ 表示空間位置，$$(\theta, \phi)$$ 表示觀察方向，輸出為一個4D向量$$(R, G, B, \sigma)$$，代表RGB顏色和體積密度。 NeRF在神經渲染和視圖合成的定量基準測試和定性測試中都取得了最新成果。

### 先前工作

#### 用於視圖合成的體積渲染

體積渲染透過使用一組影像來學習三維離散體積表示，模型估計三維空間中每個點的體積密度和發射顏色，然後用於從各種視點合成影像。先前的方法包括 **Soft 3D**，它使用傳統的立體方法實現場景的軟三維表示，直接用於在視圖合成過程中建模射線可見性和遮蔽。還有深度學習方法如 **Neural Volumes**，使用編碼器-解碼器網路將輸入圖像轉換為三維體素網格，用於生成新視圖 。雖然這些體積表示易於透過訓練渲染真實視圖來優化，但隨著場景解析度或複雜度的增加，儲存這些離散表示所需的計算和記憶體變得不切實際。

#### 作為形狀表示的神經網絡

該研究領域旨在使用神經網路的權重隱式地表示三維表面。與體積方法相比，這種表示在無限分辨率下編碼了三維表面的描述，而不會佔用過多的記憶體 。神經網路透過學習將空間中的點映射到該點的屬性來編碼三維表面，例如佔用率 或有符號距離場。雖然這種方法顯著節省了內存，但更難以優化，導致合成視圖品質較離散表示差。

### NeRF的演算法原理

NeRF將上述兩種方法結合，透過將場景表示為MLP的權重，同時使用傳統的體積渲染技術進行視圖合成。

#### 神經輻射場的場景表示

場景由一個5D向量$$(\mathbf{x}, \mathbf{d})$$ 組成，其中$$\mathbf{x} = (x, y, z)$$ 表示空間座標，$$\mathbf{d} = ( \theta, \phi)$$ 表示視角方向。這個連續的5D場景表示被一個MLP網路$$F_{\Theta} : (\mathbf{x}, \mathbf{d}) \to (\mathbf{c}, \sigma)$$ 近似，權重$$\Theta$$被最佳化以預測每個5D輸入的$$\mathbf{c} = (R, G, B)$$（RGB顏色）和$$\sigma$$（密度）。密度可以視為遮蔽程度，遮蔽較高的點具有較高的 $$\sigma$$ 值。

為了保持一致性，網路被設計為僅將$$\sigma$$ 作為$$\mathbf{x}$$ 的函數進行預測，因為密度不應隨著視角變化而變化，而$$\mathbf{c}$$ 則被訓練為$$\mathbf{x}$$ 和$$\mathbf{d}$$ 的函數。 MLP $$F_{\Theta}$$ 有9個全連接層，使用ReLU啟動函數，前8層每層有256個通道，最後一層有128個通道。 $$F_{\Theta}$$ 先處理 $$\mathbf{x}$$，透過前8層輸出 $$\sigma$$ 和一個256維的特徵向量 $$\mathbf{v}$$。然後將 $$\mathbf{v}$$ 與 $$\mathbf{d}$$ 連接，傳遞到最後一層輸出 $$\mathbf{c}$$。

#### 使用輻射場的體積渲染

任何穿過場景的光線的顏色都使用經典的體積渲染原理來渲染：

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} w_i c_i, \quad \text{其中} \quad w_i = T_i \alpha_i
$$

公式可以解釋為，每個點的顏色 $$c_i$$ 由權重 $$w_i$$ 加權，$$w_i$$ 由以下組成：

- 累計透射率$$T_i = \exp\left( -\sum_{j=1}^{i-1} \sigma_j \delta_j \right)$$，$$\sigma_j$$ 是密度，$$\delta_j$$ 是相鄰取樣點之間的距離。
- 不透明度 $$\alpha_i = 1 - \exp(-\sigma_i \delta_i)$$。

因此，在表面開始處具有較高透射率和不透明度的點對光線 $$\mathbf{r}$$ 的最終預測顏色貢獻較大。

#### 優化NeRF

為了實現高品質的渲染，原始論文還提出了兩個關鍵技術：**位置編碼（Positional Encoding）** 和 **分層體積採樣（Hierarchical Volume Sampling）**。

##### 位置編碼

作者發現直接將 $$(x, y, z, \theta, \phi)$$ 輸入 $$F_{\Theta}$$ 會導致效能不佳。因此，他們選擇使用高頻函數將輸入映射到更高維空間，這使模型能夠更好地擬合具有高變化的資料。 $$F_{\Theta}$$ 重新定義為兩個函數的組合$$F_{\Theta} = F_{\Theta}' \circ \gamma$$，其中$$F_{\Theta}'$$ 是原始MLP，$$\ gamma$$ 定義為：

$$

\gamma(p) = \left( \sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \ cos(2^{L-1} \pi p) \right) 

$$

$$\gamma(\cdot)$$ 應用於 $$\mathbf{x}$$ 的$$(x, y, z)$$ 部分，取$$L=10$$；應用於$$(\theta, \phi)$$ 部分，取$$L=4$$。這個映射顯著提高了模型效能。

##### 分層體積採樣

空閒空間和被遮蔽的區域對NeRF的質量貢獻較小，但在均勻採樣中，它們被以相同的速率採樣。為此，作者提出了一種分層表示，透過根據樣本的預期影響分配採樣點，來提高渲染效率和品質。

這透過優化兩個網路來實現：一個「粗」網路和一個「精細」網路。粗網絡在光線上均勻採樣點，而精細網絡則根據粗網絡的結果，對重要區域進行更密集的採樣。這種方法將更多的計算資源分配給對渲染品質影響更大的區域。

### 局限性

儘管NeRF在從二維影像渲染逼真三維體積方面具有突破性能力，但其原始方法存在一些限制，包括：

- **運算效率**：最佳化單一場景需要在單一NVIDIA V100 GPU上進行10萬到30萬次迭代，耗時1-2天。這是由於渲染時對光線進行密集採樣所致。
- **泛化能力不足**：NeRF固有地對單一場景過擬合，無法在不完全重新訓練的情況下適應新場景。
- **編輯困難**：由於模型將場景表示為連續函數，不儲存明確的幾何訊息，因此修改NeRF中的內容（如移動或刪除物件）非常困難。
- **資料需求**：為了產生高品質的結果，NeRF需要大量的資料。例如，原始論文中的合成三維模型（如樂高推土機和海盜船）需要大約100張圖像，現實場景（如花朵和會議室）則需要約60張。

## Instant NGP

### 概述

**Instant Neural Graphics Primitives（Instant-NGP）** 是由 NVIDIA 研究院（NVIDIA Nvlabs）提出的一種方法，顯著降低了原始 NeRF（Neural Radiance Fields）的計算需求。它利用**多重解析度雜湊編碼（Multi-Resolution Hash Encoding）** 來改善記憶體使用，並優化三維重建效能。

### 先前工作

#### 可學習的位置編碼

**可學習的位置編碼（Learnable Positional Encoding）** 是指針對連續三維空間中特定位置進行參數化的位置編碼。對於三維空間中的一個點 $$\mathbf{p}$$，其位置編碼可以表示為：

$$
\mathbf{pe}(\mathbf{p}) = \sigma(\mathbf{W} \mathbf{p} + \mathbf{b})
$$

其中，$$\mathbf{p} = (x, y, z)^\top$$ 表示三維空間中位置的座標，$$\mathbf{W}$$ 是可學習的權重矩陣，$$\mathbf{b}$$ 是偏移向量，$$\sigma$$ 是非線性激活函數。

這些位置編碼可以整合到神經網路模型中，以促進空間關係的學習。

*在原始論文的實驗中，隨著用於學習位置編碼的參數數量增加，生成的圖像變得更加清晰和銳利。 *

### 演算法

#### 多重解析度雜湊編碼

Instant-NGP 的核心組件之一是**多重解析度雜湊編碼**。與其為整個三維空間學習位置編碼，不如先將三維空間縮放至 0 到 1 的標準化範圍內。然後，在多個解析度上複製該標準化空間，每個解析度的空間被細分為不同密度的網格。

這樣可以同時捕捉場景中的粗略和精細細節。每個層級都專注於學習網格頂點的位置編碼。數學上，可以表示為：

$$
\mathbf{p}_{\text{scaled}} = \mathbf{p} \cdot \mathbf{s}
$$

其中，$$\mathbf{p}$$ 表示三維空間中的原始座標，$$\mathbf{s}$$ 是將空間標準化到$[0, 1]\) 範圍的縮放因子。完成縮放後，使用空間雜湊函數將座標雜湊到多重解析度結構中：

$$
h(\mathbf{x}) = \left( \bigoplus_{i=1}^d (x_i \cdot \pi_i) \right) \bmod T
$$

這裡，$$d$$ 是空間的維度（例如，三維座標中的$$d = 3$$），$$\mathbf{x} = (x_1, x_2, \dots, x_d)$$ 表示縮放後的座標，$$\bigoplus $$ 表示位元異或運算，$$\pi_i$$ 是每個維度獨有的大質數，$$T$$ 是雜湊表的大小。

此函數將空間座標映射到雜湊表中的索引位置，在這些位置儲存或檢索神經網路的參數，將特定的空間位置與神經網路參數關聯。

簡而言之，每個解析度層級都分配了一個哈希表。對於每個分辨率，每個網格頂點都被映射到該分辨率的哈希表中的一個條目。高分辨率的哈希表相比低分辨率的哈希表更大。每個解析度的雜湊表將其頂點映射到一組獨立的參數，用於學習它們的位置編碼。

#### 學習位置編碼

在訓練過程中，當模型從不同視角接收影像時，神經網路調整儲存在雜湊表中的參數，以最小化渲染影像與實際訓練影像之間的差異。損失函數 L$$ 可以表示為：

$$
L(\theta) = \frac{1}{2} \sum_{i=1}^m \left( R(\mathbf{x}_i, \theta) - y_i \right)^2
$$

其中，$$R(\mathbf{x}_i, \theta)$$ 是基於參數$$\theta$$ 和視角$$\mathbf{x}_i$$ 渲染的圖像，$$y_i$$ 是對應的實際圖像，$$m$$ 是考慮的像素或數據點數量。然後，我們可以使用梯度下降等最佳化技術來學習這些參數。

#### 哈希衝突的處理

為了避免哈希衝突，每個解析度都分配了一個足夠大的雜湊表，確保從條目到位置編碼的映射是唯一的。一旦發生哈希衝突，可以透過增加哈希表的大小或使用更複雜的雜湊函數來減少衝突的可能性。

### 效能表現

在保持品質的同時，Instant-NGP 相比原始的 NeRF 實現，取得了 20 到 60 倍的速度提升。

*圖表描述：原始論文中的圖表比較了各種 NeRF 實現的峰值信噪比（PSNR）性能，包括作者的多分辨率哈希編碼方法，與需要數小時訓練的其他模型進行比較。第一行是建構的物件名稱。 *

### 局限性

儘管 Instant-NGP 在加速 NeRF 的計算和訓練過程中表現出色，但它仍然存在一些問題，例如在不同資料集或未見過的場景下的泛化能力不足。

在下一節中，我們將介紹基於**潛在擴散模型（Latent Diffusion Models，LDM）** 的三維重建技術，以解決泛化能力的問題。

## Latent Diffusion Model（LDM）在三維重建的應用

### 背景

傳統的三維重建演算法高度依賴訓練資料來捕捉物體的所有細節。然而，人類能夠僅憑一張圖像就估計出三維表面。這個概念是 **Zero-1-to-3** 框架的基礎，該框架由哥倫比亞大學開發，提出了一種基於擴散模型的三維重建方法。 Zero-1-to-3 利用最初為文字生成影像設計的 **潛在擴散模型（Latent Diffusion Model，LDM）**，根據相機的外部參數（如旋轉和平移）生成影像的新視角。透過利用大規模LDM 學習到的幾何先驗，Zero-1-to-3 能夠從單張圖像生成新視角，展示了強大的零樣本泛化能力，在單視圖三維重建和新視圖合成任務中都優於先前的模型。如圖所示：

![Zero-1-to-3 框架概覽](https://Kenn3o3.github.io/files/3DGS/kenny_folder/zero-1-2-3.png)
*圖 1：Zero-1-to-3 框架概覽（來自原始論文）*

### 去噪擴散機率模型（DDPM）

**去噪擴散機率模型（Denoising Diffusion Probabilistic Models，DDPM）** 是一類生成模型，透過在一系列步驟中逐漸添加雜訊來轉換數據，然後學習逆過程以從雜訊中產生新樣本。

#### 前向過程

DDPM 的前向過程是一個馬可夫鏈，在 $$T$$ 個時間步中逐漸在資料中加入高斯雜訊。數學描述如下：

$$
x_{t} = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中，$$x_0$$ 是原始數據，$$x_t$$ 是第 $$t$$ 個時間步的數據，$$\epsilon$$ 是噪聲，$$\alpha_t$$ 是方差調度參數，決定每一步添加的噪聲量。

![DDPM 的前向過程](https://Kenn3o3.github.io/files/3DGS/kenny_folder/ddpm_forward_process.png)
*圖 2：DDPM 的前向過程（https://www.youtube.com/watch?v=ifCDXFdeaaM\&t=608）*

#### 逆向流程

逆向過程旨在透過學習參數化模型 $$p_{\theta}$$，從雜訊重建原始資料。逆向過程也是馬可夫鏈，描述如下：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_ {\theta}(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

其中，$$\epsilon_{\theta}(x_t, t)$$ 是用於預測雜訊的神經網絡，$$\sigma_t$$ 是逆過程雜訊的標準差，$$\bar{\alpha}_t = \prod_{s= 1}^t \alpha_s$$ 是$$\alpha_t$$ 的累積乘積。

#### 訓練過程

訓練 DDPM 涉及優化神經網路的參數 $$\theta$$，以最小化模型預測的雜訊與實際添加的雜訊之間的差異。損失函數通常是兩個雜訊項之間的均方誤差：

$$
L(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[ \| \epsilon - \epsilon_{\theta}(x_t, t) \|_2^2 \right]
$$

#### 取樣過程

為了產生新樣本，逆向過程從純雜訊 $$x_T \sim \mathcal{N}(0, I)$$ 開始，迭代應用逆向步驟，以逼近原始資料分佈 $$x_0$$。

**DDPM 採樣演算法：**

1. 初始化：從 $$\mathcal{N}(0, I)$$ 取樣$$ x_T$$。
2. 對於$$t = T$$ 到 1：
   - 計算$$\bar{\alpha}_t$$ 和$$\sigma_t^2$$。
   - 使用神經網路預測雜訊$$\epsilon_{\theta}(x_t, t)$$。
   - 更新$$x_{t-1}$$。
3. 返回$$x_0$$。

![DDPM 的採樣過程](https://Kenn3o3.github.io/files/3DGS/kenny_folder/ddpm_sampling.png)
*圖 3：DDPM 的取樣過程（https://www.youtube.com/watch?v=ifCDXFdeaaM\&t=608）*

### Zero-1-to-3 中的潛在擴散模型（LDM）

**潛在擴散模型（Latent Diffusion Models，LDM）** 是一種結合了擴散模型和變分自編碼器（VAE）優勢的生成模型。傳統的 DDPM 在像素空間操作，計算量大。 LDM 將影像資料壓縮到潛在空間，再在潛在空間中進行擴散和去噪，提高了產生高品質影像的效率。

![LDM 架構](https://Kenn3o3.github.io/files/3DGS/kenny_folder/LDM.png)
*圖 4：LDM 的架構（來自原始論文）*

#### 訓練 LDM 模型$$\epsilon_{\theta}$$

LDM 的訓練包括兩個主要階段：

1. **訓練 VAE：** 學習編碼函數$$E(x)$$ 和解碼函數$$D(z)$$，將高解析度影像$$x$$ 壓縮為潛在表示$$z$$，並嘗試重建。

 損失函數為：

 $$
 \mathcal{L}_{\text{VAE}}(\phi, \psi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\psi(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))
 $$

 ![訓練 VAE](https://Kenn3o3.github.io/files/3DGS/kenny_folder/training_VAE.png)
 *圖 5：訓練 VAE（圖片來自 Lightning AI）*

2. **訓練 Attention-U-Net 去噪器：** 在潛在空間中訓練去噪模型，學習將雜訊樣本$$z_T$$ 轉換為資料分佈$$z_0$$。

 損失函數為：

 $$
 \mathcal{L}(\theta) = \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0, I), t} \left[ \| \epsilon - \epsilon_{\theta} (z_t, t) \|^2 \right]
 $$

 ![最小化分佈差異](https://Kenn3o3.github.io/files/3DGS/kenny_folder/dis.png)
 *圖 6：最小化分佈差異（取自 https://agustinus.kristia.de/techblog/2016/12/21/forward-reverse-kl/ ）*

 ![Attention-U-Net 架構](https://Kenn3o3.github.io/files/3DGS/kenny_folder/attention_u_net.png)
 *圖 7：Attention-U-Net 架構（來自原始論文）*

 ![訓練 Attention-U-Net 去雜訊](https://Kenn3o3.github.io/files/3DGS/kenny_folder/training_unet.png)
 *圖 8：訓練 Attention-U-Net 去雜訊（圖片來自 Lightning AI）*

#### 基於相機參數的條件生成

Zero-1-to-3 框架的關鍵在於根據相機的外部參數（旋轉$$R$$ 和平移$$t$$）對 LDM 進行條件化，以產生新視角。

- **相機變換：** 調整潛在變數$$z$$ 以反映視角變化：

 $$
 z' = f(z, R, t)
 $$

- **模型適應：** 擴展 LDM 訓練，不僅從$$z$$ 生成圖像，還從變換後的$$z'$$ 產生新視角的圖像。

- **損失函數：** 計算輸出影像與真實影像之間的均方誤差（MSE），並使用「近視圖一致性損失」來保持三維重建的一致性。

#### 三維重建過程

透過對不同的$$R$$ 和$$t$$ 應用上述方法，產生物體的多個視角。然後，將這些影像整合：

$$
\text{3D 模型} = \text{整合}(\{x'\})
$$

整合過程通常使用體積融合或多視圖立體演算法，建立物件的詳細三維表示。

![Zero-1-to-3 的三維重建範例](https://Kenn3o3.github.io/files/3DGS/kenny_folder/zero-1-to-3-demo.png)
*圖 9：Zero-1-to-3 的三維重建範例*

### 基於擴散模型和 NeRF 的三維重建的局限性

- **彈性與即時渲染：** 訓練 Zero-1-to-3 模型通常需要在取樣過程中進行迭代去噪，計算量大且速度慢，不適合即時渲染應用。

- **隱式表示的模糊性：** NeRF 和擴散模型都使用隱式方式表示三維物體，沒有明確建構三維空間。這雖然節省了空間，但可能導致模型解釋上的不明確。

## 3D Gaussian Splatting

### 背景

在三維場景重建的發展過程中，**明確表示**（如網格和點雲）由於其結構清晰、渲染速度快、易於編輯等優點，一直受到開發者和研究者的青睞。基於 NeRF（Neural Radiance Fields）的方法則轉向了三維場景的**連續表示**。儘管這些連續表示方法在最佳化方面有優勢，但渲染時所需的隨機取樣計算成本高，而且可能導致雜訊。此外，隱式表示缺乏明確的幾何訊息，不利於後期編輯。

### 概述

**三維高斯散射（3D Gaussian Splatting）** 提供了一種高品質的即時渲染三維場景的新視角方法。它利用高斯函數來表示平滑且精確的紋理，透過捕捉場景的照片來實現。
![3D Gaussian Splatting 過程概覽](https://Kenn3o3.github.io/files/3DGS/Olivers/image.png)

*圖 1：3D Gaussian Splatting 流程概覽*

為了使用三維高斯散射重建三維模型，首先需要從不同角度拍攝物體的視頻，然後將其轉換為代表不同相機角度的靜態場景幀。接下來，對這些圖像應用 **結構化運動恢復（Structure from Motion，SfM）**，以及諸如 SIFT 等特徵檢測和匹配技術，生成稀疏點雲。點雲中的三維資料點被表示為三維高斯分佈。透過優化過程、自適應的高斯密度控制和高效的演算法設計，可以以高幀率重建三維模型的逼真視圖。

3D Gaussian Splatting 的演算法如下所示，可分為三個部分：初始化、最佳化和高斯的密度控制。

### 演算法

#### 初始化

使用 SfM 產生的稀疏三維點雲中的點被初始化為三維高斯分佈。高斯由以下變數定義：

- 位置 $$\mathbf{p}$$
- 世界座標系下的三維協方差矩陣 $$\Sigma$$
- 不透明度 $$\alpha$$
- 球諧函數係數 $$\mathbf{c}$$（表示 RGB 顏色）

高斯函數的公式為：

$$
G(\mathbf{x}) = e^{-\frac{1}{2} \mathbf{x}^\top \Sigma^{-1} \mathbf{x}}
$$

#### 最佳化

最初，高斯分佈較為稀疏，無法充分代表場景。透過優化過程，它們逐漸調整以更好地表示場景。具體步驟如下：

1. **選擇視角**：從訓練集中隨機選擇一個相機視角 $$V$$ 及其對應的影像 $$\hat{I}$$。
2. **渲染圖像**：使用高斯的均值$$\mathbf{p}$$、協方差矩陣$$\Sigma$$、顏色係數$$\mathbf{c}$$、不透明度$$\alpha$$ 和視角$$V$$ ，透過可微的光柵化函數$$Rasterize()$$ 產生渲染影像$$I$$。
3. **計算損失**：使用如下的損失函數計算渲染影像 $$I$$ 與真實影像 $$\hat{I}$$ 之間的差異：

 $$
 \mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\textrm{D-SSIM}}
 $$

 - $$\mathcal{L}_1$$：渲染影像與真實影像的平均絕對誤差（MAE）
 - $$\mathcal{L}_{\textrm{D-SSIM}}$$：差異結構相似性指數（Difference Structural Similarity Index）
 - $$\lambda$$：調整兩個損失項權重的預定義參數

4. **參數更新**：使用 Adam 優化器根據損失的梯度更新高斯的參數。

#### 高斯的自適應密度控制

在初始化之後，採用自適應的方法來控制高斯的數量和密度，以優化靜態三維場景的表示。具體行為如下：

- **高斯移除**：每 100 次迭代，如果某個高斯在三維空間中過大或其不透明度低於閾值 $$\epsilon_\alpha$$（即幾乎透明），則將其移除。
- **高斯複製**：對於填充區域不足的高斯，如果其小於定義的閾值，則克隆該高斯並沿其方向移動以覆蓋空白區域。此操作自適應地增加高斯的數量和體積，直到該區域被良好擬合。
- **高斯分裂**：對於過度重建的區域（高斯的方差過高），將其按照因子 $$\phi$$ 分裂為更小的高斯。原論文使用 $$\phi = 1.6$$。

![高斯的密度控制](https://Kenn3o3.github.io/files/3DGS/Olivers/des.png)

*圖 2：在過度重建區域進行高斯分裂，在欠重建區域進行高斯克隆*

由於分裂和複製操作，高斯的數量會增加。為了解決這個問題，每經過$$N = 3000$$ 次迭代，對所有高斯的$$\alpha$$ 進行重置，然後在優化步驟中增加所需的高斯的$$\alpha$$ 值，允許未使用的高斯被移除。

### $$\Sigma$$ 和 $$\Sigma'$$ 的梯度計算

每個高斯在三維空間中表示為一個橢球，由協方差矩陣 $$\Sigma$$ 建模：

$$
\Sigma = R S S^\top R^\top
$$

其中，$$S$$ 是縮放矩陣，$$R$$ 是旋轉矩陣。

對於每個相機角度，三維高斯會被光柵化到二維螢幕上。透過視角轉換矩陣 $$W$$ 和雅可比矩陣 $$J$$，可以得到二維空間的協方差矩陣 $$\Sigma'$$：

$$
\Sigma' = J W \Sigma W^\top J^\top
$$

$$\Sigma'$$ 是一個 $$3 \times 3$$ 的矩陣，可以透過忽略第三行和第三列的元素，簡化為對應的二維平面表示。

為了計算三維空間協方差的梯度，應用鍊式法則，將 $$\Sigma$$ 和 $$\Sigma'$$ 對它們的旋轉 $$q$$ 和縮放 $$s$$ 求導：

$$
\frac{d\Sigma'}{ds} = \frac{d\Sigma'}{d\Sigma} \frac{d\Sigma}{ds}, \quad \frac{d\Sigma'}{dq} = \frac{d\Sigma'}{d\Sigma} \frac{d\Sigma}{dq}
$$

將 $$U = J W$$ 代入 $$\Sigma'$$ 的表達式，得到：

$$
\Sigma' = U \Sigma U^\top
$$

這樣，$$\frac{\partial \Sigma'}{\partial \Sigma_{ij}}$$ 可以表示為 $$U$$ 的函數：

$$
\frac{\partial \Sigma'}{\partial \Sigma_{ij}} = \begin{pmatrix}
U_{1i} U_{1j} & U_{1i} U_{2j} \\
U_{1j} U_{2i} & U_{2i} U_{2j}
\end{pmatrix}
$$

將 $$M = R S$$ 代入 $$\Sigma$$ 的表達式，可以改寫為 $$\Sigma = M M^\top$$。然後使用鍊式法則，得到：

$$
\frac{d\Sigma}{ds} = \frac{d\Sigma}{dM} \frac{dM}{ds}, \quad \frac{d\Sigma}{dq} = \frac{d\Sigma} {dM} \frac{dM}{dq}
$$

縮放矩陣 $$s$$ 處的梯度可以計算為：

$$
\frac{\partial M_{ij}}{\partial s_k} = \begin{cases}
R_{ik} & \text{如果 } j = k \\
0 & \text{否則}
\end{cases}
$$

為了定義 $$M$$ 關於旋轉矩陣 $$R$$ 的導數，需要將旋轉矩陣 $$R$$ 表示為四元數 $$q$$ 的函數：

$$
R(q) = 2 \begin{pmatrix}
\frac{1}{2} - (q_j^2 + q_k^2) & q_i q_j - q_r q_k & q_i q_k + q_r q_j \\
q_i q_j + q_r q_k & \frac{1}{2} - (q_i^2 + q_k^2) & q_j q_k - q_r q_i \\
q_i q_k - q_r q_j & q_j q_k + q_r q_i & \frac{1}{2} - (q_i^2 + q_j^2)
\end{pmatrix}
$$

然後，可以計算四元數 $$q_r, q_i, q_j, q_k$$ 的梯度 $$\frac{\partial M}{\partial q_x}$$。

### 基於 Tile 的光柵化器用於即時渲染

為了快速渲染高斯建構的三維模型，使用了 **基於 Tile 的光柵化器（Tile-based Rasterizer）**（見下圖）。此方法首先利用給定的相機視角 $$V$$ 和其位置 $$\mathbf{p}$$，過濾掉不在視錐內的高斯，從而僅處理對當前視角有貢獻的高斯。這種方法減少了渲染過程中需要處理的數據，並提高了渲染效率。

![基於 Tile 的光柵化器示意圖](https://Kenn3o3.github.io/files/3DGS/Olivers/tile.png)

*圖 3：基於 Tile 的光柵化器區域，其中綠色高斯對視錐有貢獻，紅色高斯無貢獻，給定相機角度 $$V$$*

與其對每個像素進行 $$\alpha$$ 混合，不如將螢幕分成 16×16 的小塊（Tile）。每個高斯根據其視角深度和所在的 Tile 被指派一個實例鍵（Instance Key），然後使用 GPU 的 Radix 排序演算法對高斯進行排序。排序後的高斯列表可以反映 Tile 的深度層次。

排序完成後，為每個 Tile 分配一個線程區塊，將對應的高斯載入到共享記憶體中。然後，從前到後對排序後的高斯列表進行 $$\alpha$$ 混合，直到每個像素達到目標的 alpha 飽和度。這種設計最大化了運算效率，利用了 Tile 渲染和高斯載入的平行性。

### 局限性

#### 大記憶體頻寬需求

為了實現高幀率的即時渲染，在 $$Rasterize()$$ 函數中使用了平行計算的方法。這涉及大量動態資料在共享記憶體中的載入和處理，需要較大的 GPU 記憶體頻寬來支援資料傳輸。

#### 在稀疏視角下的穩健性

高斯的優化是透過與真實相機視角的梯度比較進行的。然而，對於包含較少或沒有資料點的視角，由於缺乏足夠的資料來優化該區域的高斯，可能會導致渲染結果出現偽影和失真。

# 3D重建的未來趨勢

## 以語義為導向的3D重建 (Semantic-Driven 3D Reconstruction)
目前大多數的3D重建技術專注於從影像中生成3D模型。然而，整合文字提示作為引導因素為未來研究開啟了令人興奮的可能性。例如，論文《FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models》 展示了如何利用文本線索顯著提升重建模型的精確性與上下文相關性。此外，像《LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation》中提出的方法，更是展示了僅依靠文字提示進行零樣本（Zero-Shot）的高解析度3D內容生成的潛力。

## 動態3D場景重建 (Dynamic 3D Scene Reconstruction)
現有的技術主要針對靜態場景進行3D模型的重建，這意味著場景在捕捉過程中如果有結構變化，可能會導致特定區域的重建不足。為解決此問題，《4D Gaussian Splatting for Real-Time Dynamic Scene Rendering》 提出了以一組標準3D Gaussian分佈為基礎，通過形變場 (Deformation Field) 在不同行為時間點進行轉換的方法。此技術能夠生成動態變化的3D模型，有效表示物體的運動過程，實現動態3D場景重建。

## 單視角3D重建 (Single View 3D Reconstruction)
基於《Zero 1-to-3》的方法論，單視角3D重建逐漸成為一個受到關注的研究領域。此技術結合了擴散模型 (Diffusion Models)，從單張圖像生成3D物體。相關研究如LGM和 InstantMesh，在此領域展示了極具潛力的成果，為單一視角下的3D生成帶來了嶄新的可能性。

---

# Reference

1. **Jonathan Ho**, **Ajay Jain**, and **Pieter Abbeel**. *Denoising Diffusion Probabilistic Models*, 2020.

2. **Bernhard Kerbl**, **Georgios Kopanas**, **Thomas Leimkühler**, and **George Drettakis**. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, 2023.

3. **Agustinus Kristiadi**. *KL Divergence: Forward vs Reverse?* [https://agustinus.kristia.de/techblog/2016/12/21/forward-reverse-kl/](https://agustinus.kristia.de/techblog/2016/12/21/forward-reverse-kl/), 2016. Accessed: 2024-04-22.

4. **Hung-yi Lee**. *Forward Process of DDPM*, April 2023. [https://www.youtube.com/watch?v=ifCDXFdeaaM&t=608](https://www.youtube.com/watch?v=ifCDXFdeaaM&t=608).

5. **Ruoshi Liu**, **Rundi Wu**, **Basile Van Hoorick**, **Pavel Tokmakov**, **Sergey Zakharov**, and **Carl Vondrick**. *Zero-1-to-3: Zero-Shot One Image to 3D Object*, 2023.

6. **Stephen Lombardi**, **Tomas Simon**, **Jason Saragih**, **Gabriel Schwartz**, **Andreas Lehrmann**, and **Yaser Sheikh**. *Neural Volumes: Learning Dynamic Renderable Volumes from Images*. **ACM Transactions on Graphics**, 38(4):65:1–65:14, July 2019.

7. **Ricardo Martin-Brualla**, **Noha Radwan**, **Mehdi S. M. Sajjadi**, **Jonathan T. Barron**, **Alexey Dosovitskiy**, and **Daniel Duckworth**. *NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections*, 2021.

8. **Lars Mescheder**, **Michael Oechsle**, **Michael Niemeyer**, **Sebastian Nowozin**, and **Andreas Geiger**. *Occupancy Networks: Learning 3D Reconstruction in Function Space*, 2019.

9. **Ben Mildenhall**, **Pratul P. Srinivasan**, **Matthew Tancik**, **Jonathan T. Barron**, **Ravi Ramamoorthi**, and **Ren Ng**. *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, 2020.

10. **Thomas Müller**, **Alex Evans**, **Christoph Schied**, and **Alexander Keller**. *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*. **ACM Transactions on Graphics**, 41(4):1–15, July 2022.

11. **Jeong Joon Park**, **Peter Florence**, **Julian Straub**, **Richard Newcombe**, and **Steven Lovegrove**. *DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation*, 2019.

12. **Eric Penner** and **Li Zhang**. *Soft 3D Reconstruction for View Synthesis*, 36(6), 2017.

13. **Robin Rombach**, **Andreas Blattmann**, **Dominik Lorenz**, **Patrick Esser**, and **Björn Ommer**. *High-Resolution Image Synthesis with Latent Diffusion Models*, 2022.

14. **Jiaxiang Tang**, **Zhaoxi Chen**, **Xiaokang Chen**, **Tengfei Wang**, **Gang Zeng**, and **Ziwei Liu**. *LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation*. **arXiv preprint arXiv:2402.05054**, 2024.

15. **Jonathan Tremblay**, **Moustafa Meshry**, **Alex Evans**, **Jan Kautz**, **Alexander Keller**, **Sameh Khamis**, **Thomas Müller**, **Charles Loop**, **Nathan Morrical**, **Koki Nagano**, **Towaki Takikawa**, and **Stan Birchfield**. *RTMV: A Ray-Traced Multi-View Synthetic Dataset for Novel View Synthesis*, 2022.

16. **Guanjun Wu**, **Taoran Yi**, **Jiemin Fang**, **Lingxi Xie**, **Xiaopeng Zhang**, **Wei Wei**, **Wenyu Liu**, **Qi Tian**, and **Xinggang Wang**. *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*, 2023.

17. **Jiale Xu**, **Weihao Cheng**, **Yiming Gao**, **Xintao Wang**, **Shenghua Gao**, and **Ying Shan**. *InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-View Large Reconstruction Models*, 2024.

18. **Hao Zhang**, **Yanbo Xu**, **Tianyuan Dai**, **Yu-Wing Tai**, and **Chi-Keung Tang**. *FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models*, 2023.

19. **Qiang Zhang**, **Seung-Hwan Baek**, **Szymon Rusinkiewicz**, and **Felix Heide**. *Differentiable Point-Based Radiance Fields for Efficient View Synthesis*, 2023.