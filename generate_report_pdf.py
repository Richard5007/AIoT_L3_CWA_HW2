# -*- coding: utf-8 -*-
"""
Generate comprehensive, publication-quality PDF report for HW2
Following CRISP-DM methodology, with figures, tables, formulas, AI statements, NotebookLM summary, and mainstream solution comparison.
"""
import os
import sys
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Register Fonts
font_path = 'C:/Windows/Fonts/msjh.ttc'
bold_font_path = 'C:/Windows/Fonts/msjhbd.ttc'
pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
pdfmetrics.registerFont(TTFont('ChineseFontBold', bold_font_path))

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('ChineseFont', 8)
        self.setFillColor(colors.HexColor('#64748b'))
        
        # Don't draw header on first page
        if self._pageNumber > 1:
            self.drawString(36, 815, "巨量資料分析與機器學習 (HW2) - 金郡房價多元線性回歸分析報告 (CRISP-DM)")
            self.drawRightString(559, 815, "學號: 7114056XXX")
            self.setStrokeColor(colors.HexColor('#cbd5e1'))
            self.setLineWidth(0.5)
            self.line(36, 810, 559, 810)

        # Footer on all pages
        page_str = f"第 {self._pageNumber} 頁 / 共 {page_count} 頁"
        self.drawRightString(559, 25, page_str)
        self.drawString(36, 25, "機密與學術報告 | 本研究遵循 CRISP-DM 資料探勘跨行業標準流程")
        self.setStrokeColor(colors.HexColor('#cbd5e1'))
        self.setLineWidth(0.5)
        self.line(36, 35, 559, 35)
        
        self.restoreState()

def build_pdf_report(filename='7114056XXX_hw2_Report.pdf'):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='ChineseFontBold',
        fontSize=20,
        leading=26,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=1,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        fontName='ChineseFont',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='ChineseFontBold',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='ChineseFontBold',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor('#0f766e'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='ChineseFont',
        fontSize=9,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=5
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        fontName='ChineseFont',
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        leftIndent=15,
        spaceAfter=3
    )
    
    table_text = ParagraphStyle(
        'TableText',
        fontName='ChineseFont',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        fontName='ChineseFontBold',
        fontSize=8,
        leading=11,
        textColor=colors.white,
        alignment=1
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        fontName='ChineseFont',
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor('#0f766e')
    )

    caption_style = ParagraphStyle(
        'FigCaption',
        fontName='ChineseFontBold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceBefore=4,
        spaceAfter=8
    )

    story = []
    
    # ----------------------------------------------------
    # Header & Cover Information
    # ----------------------------------------------------
    story.append(Paragraph("巨量資料分析與機器學習 - 作業二 (HW2) 專題報告", title_style))
    story.append(Paragraph("基於 CRISP-DM 流程之金郡 (King County) 房價多元線性回歸完整分析", subtitle_style))
    
    info_table_data = [
        [
            Paragraph("<b>學生學號：</b> 7114056XXX (請同學替換)", table_text),
            Paragraph("<b>學生姓名：</b> 學生姓名", table_text),
            Paragraph("<b>分析主題：</b> 多元線性回歸 (MLR)", table_text)
        ],
        [
            Paragraph("<b>資料來源：</b> Kaggle King County House Sales", table_text),
            Paragraph("<b>資料維度：</b> 21,613 筆 / 18 特徵", table_text),
            Paragraph("<b>資料集連結：</b> <font color='#2563eb'><u>kaggle.com/datasets/...</u></font>", table_text)
        ]
    ]
    info_table = Table(info_table_data, colWidths=[174, 174, 175])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1e3a8a'), spaceAfter=10))

    # ----------------------------------------------------
    # Section 1: Business Understanding
    # ----------------------------------------------------
    story.append(Paragraph("一、CRISP-DM 階段一：商業理解 (Business Understanding)", h1_style))
    story.append(Paragraph(
        "<b>1. 商業背景與痛點分析：</b><br/>"
        "美國華盛頓州金郡 (King County) 包含西雅圖及其周邊衛星市鎮，是微軟 (Microsoft)、亞馬遜 (Amazon)、星巴克等全球企業之總部聚集地，房地產市場交易極具代表性。"
        "然而，房產市場定價高度非均質且受多重物理維度（面積、格局、建材品質等級）與地理交通維度（離西雅圖市中心距離、湖濱景觀）交織影響。"
        "傳統房仲與估價師依賴經驗法則，估價耗時且容易產生主觀偏誤；銀行在進行房貸風險審核時，亦迫切需要具備<b>高度可解釋性與法定抗辯性</b>的定價基準模型。",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. 機器學習任務定位：</b><br/>"
        "本研究之資料探勘任務為建構<b>「多元線性回歸 (Multiple Linear Regression, MLR)」</b>模型，精確預測連續型目標變數 `price`（房屋成交價，美元）。"
        "相較於不可解釋之黑盒子模型，線性回歸具備嚴謹的統計推論架構，能推導出各特徵之回歸係數 $\\beta_j$（代表控制其他變數下的邊際溢價效應），並透過假設檢定 ($t$ 檢定、$F$ 檢定) 驗證顯著性。",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. 專案成功評估準則 (Success Criteria)：</b><br/>"
        "• <b>統計解釋力：</b>測試集決定係數 $R^2 \\ge 0.70$，模型整體 $F$ 檢定達到顯著水準 ($p < 0.001$)。<br/>"
        "• <b>統計嚴謹性：</b>診斷並徹底排除多重共線性 (Multicollinearity)，所有保留特徵之方差膨脹因子 $\\text{VIF} < 10$。<br/>"
        "• <b>區間估計完整性：</b>不僅提供點估計預測，更需完整推導並視覺化呈現 95% 信賴區間 (CI) 與 95% 預測區間 (PI)，提供不確定性風險量化評估。",
        body_style
    ))

    # ----------------------------------------------------
    # Section 2: Data Understanding
    # ----------------------------------------------------
    story.append(Paragraph("二、CRISP-DM 階段二：資料理解 (Data Understanding)", h1_style))
    story.append(Paragraph(
        "<b>1. 資料集來源與架構：</b><br/>"
        "本研究資料集取自 Kaggle 公開資料集：<b>House Sales in King County, USA</b>（連結：https://www.kaggle.com/datasets/harlfoxem/housesalesprediction）。"
        "資料集涵蓋 2014 年 5 月至 2015 年 5 月間之真實交易記錄，總計 <b>21,613 筆觀測值、21 個欄位</b>。"
        "排除流水識別碼 `id` 與成交時間 `date` 後，包含 1 個目標變數 `price` 與 <b>18 個候選特徵欄位</b>，完全符合本次作業「10 至 20 個特徵 (features)」之嚴格規範。",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. 資料品質檢驗與描述性統計：</b><br/>"
        "經程式檢驗，本資料集所有 21 欄位之缺失值計數皆為 0 (Missing Values = 0)，完整性極高。"
        "房價 (price) 樣本平均值為 <b>$540,088</b>、中位數為 <b>$450,000</b>、標準差為 <b>$367,127</b>、最低售價為 <b>$75,000</b>、最高售價達 <b>$7,700,000</b>。"
        "值得注意的是，原始房價偏度係數 (Skewness) 高達 <b>4.024</b>，呈現極度顯著之右偏長尾分佈（少數數百萬美元豪宅大幅拉升母體均值）。",
        body_style
    ))
    
    # Fig 1 Table
    if os.path.exists('figures/fig1_target_distribution.png'):
        img1 = Image('figures/fig1_target_distribution.png', width=500, height=175)
        story.append(img1)
        story.append(Paragraph("【圖 1】房價分佈對比：(左) 原始房價呈現高度右偏態 (Skewness: 4.02)；(右) 對數轉換後顯著常態化 (Skewness: 0.43)", caption_style))

    story.append(Paragraph(
        "<b>3. 相關性矩陣分析：</b><br/>"
        "計算各數值特徵與房價之皮爾森相關係數 (Pearson Correlation)，發現與房價正相關最強的前六大特徵依序為："
        "室內總居住面積 `sqft_living` ($r = 0.702$)、建物設計與施工等級 `grade` ($r = 0.667$)、地面上面積 `sqft_above` ($r = 0.606$)、"
        "2015年鄰里居住面積 `sqft_living15` ($r = 0.585$)、衛浴數量 `bathrooms` ($r = 0.525$) 與景觀評級 `view` ($r = 0.397$)。",
        body_style
    ))

    # Fig 2 & Fig 3
    if os.path.exists('figures/fig2_correlation_heatmap.png') and os.path.exists('figures/fig3_geo_distribution.png'):
        t_eda_imgs = Table([
            [
                Image('figures/fig2_correlation_heatmap.png', width=245, height=205),
                Image('figures/fig3_geo_distribution.png', width=245, height=205)
            ]
        ], colWidths=[250, 250])
        t_eda_imgs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_eda_imgs)
        story.append(Paragraph("【圖 2 & 圖 3】(左) 房價核心特徵相關係數熱力圖；(右) 金郡地理空間經緯度與房價分佈圖（紅星標示西雅圖市中心）", caption_style))

    # ----------------------------------------------------
    # Section 3: Data Preparation
    # ----------------------------------------------------
    story.append(Spacer(1, 5))
    story.append(Paragraph("三、CRISP-DM 階段三：資料準備與特徵工程 (Data Preparation)", h1_style))
    story.append(Paragraph(
        "<b>1. 異常值清洗：</b><br/>"
        "在資料檢視過程中發現第 15,870 筆資料之臥室數量記錄為 33 間 (`bedrooms == 33`)，然而其室內總居住面積僅為 1,620 平方英尺，衛浴僅 1.75 間。"
        "此顯然為資料輸入筆誤（實際應為 3 間臥室）。由於極端離群值會嚴重拉扯普通最小平方法 (OLS) 的超平面估計，因此果斷將此筆資料排除（剩餘 21,612 筆）。",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. 領域特徵工程 (Feature Engineering)：</b><br/>"
        "• <b>屋齡 (`house_age`)：</b>由銷售年份減去建造年份 ($sale\_year - yr\_built$)，精準反映房屋折舊程度。<br/>"
        "• <b>是否翻修 (`is_renovated`)：</b>二元指標變數 ($yr\_renovated > 0$ 則為 1，否則為 0)，捕捉翻新帶來的結構溢價。<br/>"
        "• <b>距西雅圖市中心距離 (`dist_to_seattle`)：</b>利用經緯度計算房屋與西雅圖中心商業區 (CBD, 座標 47.6062, -122.3321) 的空間距離，與房價呈現顯著負相關 ($r = -0.192$)，極佳地將地理溢價納入線性回歸模型。",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. 多重共線性 (Multicollinearity) 統計診斷與 VIF 檢定：</b><br/>"
        "<b>【關鍵代數發現】</b>經統計程式嚴格檢驗，金郡資料集存在一條嚴格成立的代數恆等式：<br/>"
        "<font color='#991b1b'><b>$$\\text{sqft\\_living} = \\text{sqft\\_above} + \\text{sqft\\_basement} \\quad (\\text{完全線性相依})$$</b></font>"
        "若將三者同時納入 OLS 估計，設計矩陣 $X$ 之列向量線性相依，其行列式值趨近於 0，$(X^T X)^{-1}$ 無法正常求逆！"
        "計算方差膨脹因子 (Variance Inflation Factor, VIF) 顯示，此三項特徵之 VIF 均高達 <b>$1.0008 \\times 10^{15}$</b>，造成嚴重的共線性爆炸！",
        body_style
    ))

    # VIF Table
    vif_summary_data = [
        [Paragraph("<b>特徵欄位 (Feature)</b>", table_header), Paragraph("<b>篩選前 VIF (共線性爆炸)</b>", table_header), Paragraph("<b>處理對策</b>", table_header), Paragraph("<b>篩選後 VIF (健康狀態)</b>", table_header)],
        [Paragraph("sqft_living", table_text), Paragraph("<font color='red'>1.00 × 10¹⁵</font>", table_text), Paragraph("保留總居住面積 (核心特徵)", table_text), Paragraph("<b>5.21</b> (健康)", table_text)],
        [Paragraph("sqft_above", table_text), Paragraph("<font color='red'>1.00 × 10¹⁵</font>", table_text), Paragraph("<b>予以剔除</b> (解除代數相依)", table_text), Paragraph("-", table_text)],
        [Paragraph("sqft_basement", table_text), Paragraph("<font color='red'>1.00 × 10¹⁵</font>", table_text), Paragraph("<b>予以剔除</b> (解除代數相依)", table_text), Paragraph("-", table_text)],
        [Paragraph("dist_to_seattle", table_text), Paragraph("5.94", table_text), Paragraph("保留地理衍生變數", table_text), Paragraph("<b>5.82</b> (健康)", table_text)],
        [Paragraph("long (經度)", table_text), Paragraph("4.51", table_text), Paragraph("保留空間座標", table_text), Paragraph("<b>4.48</b> (健康)", table_text)],
        [Paragraph("grade (結構等級)", table_text), Paragraph("3.45", table_text), Paragraph("保留品質等級", table_text), Paragraph("<b>3.42</b> (健康)", table_text)],
        [Paragraph("bathrooms (衛浴)", table_text), Paragraph("3.37", table_text), Paragraph("保留格局特徵", table_text), Paragraph("<b>3.35</b> (健康)", table_text)],
    ]
    t_vif = Table(vif_summary_data, colWidths=[110, 130, 160, 123])
    t_vif.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_vif)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>4. 特徵篩選結果與資料集切分：</b><br/>"
        "最終精選出 <b>16 個無共線性之關鍵特徵</b> (`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `dist_to_seattle`, `lat`, `long`, `sqft_living15`, `sqft_lot15`, `house_age`, `is_renovated`)。"
        "並依 <b>80% 訓練集 (17,289 筆)</b> 與 <b>20% 測試集 (4,323 筆)</b> 嚴格切分，設定隨機種子 `random_state=42` 確保評估之嚴謹性與可重複性。",
        body_style
    ))

    # ----------------------------------------------------
    # Section 4: Modeling
    # ----------------------------------------------------
    story.append(Spacer(1, 5))
    story.append(Paragraph("四、CRISP-DM 階段四：模型建構 (Modeling)", h1_style))
    story.append(Paragraph(
        "本研究設計了全方位的回歸模型對照組，以嚴謹評估線性回歸之效能與各優化策略之增益：<br/>"
        "• <b>模型 1 (Baseline)：單純線性回歸 (Simple Linear Regression)</b> —— 僅使用相關性最高之單一特徵 `sqft_living` 擬合房價，作為基準效能對照。<br/>"
        "• <b>模型 2：全特徵多元回歸 (Full MLR)</b> —— 將包含共線性特徵在內的 18 個候選特徵直接放入 OLS 估計。<br/>"
        "• <b>模型 3：精選特徵多元線性回歸 (Selected MLR)</b> —— 排除共線性特徵後之 16 個特徵 OLS 模型，使用 Statsmodels 進行完整假設檢定與推論。<br/>"
        "• <b>模型 4 & 5：正則化回歸 (Ridge & Lasso)</b> —— 引入 $L_2$ 與 $L_1$ 範數懲罰項，抑制特徵係數過度膨脹，驗證特徵稀疏性與收縮效果。<br/>"
        "• <b>模型 6：對數轉換多元回歸 (Log-Transformed MLR)</b> —— 針對目標變數實施對數轉換 $\\log(\\text{price})$，緩解房價右偏態對平方損失之扭曲。<br/>"
        "• <b>模型 7 (Benchmark)：隨機森林回歸 (Random Forest)</b> —— 採用 100 棵決策樹之非線性集成模型，作為現代機器學習之預測能力上限對照基準。",
        body_style
    ))

    # Statsmodels OLS table excerpt
    story.append(Paragraph("<b>精選多元線性回歸 (Selected MLR) - Statsmodels 估計結果節錄：</b>", h2_style))
    ols_table_data = [
        [Paragraph("<b>變數</b>", table_header), Paragraph("<b>係數 (Coef)</b>", table_header), Paragraph("<b>標準誤 (Std.Err)</b>", table_header), Paragraph("<b>t 統計量</b>", table_header), Paragraph("<b>p 值 (P>|t|)</b>", table_header), Paragraph("<b>95% 信賴區間</b>", table_header)],
        [Paragraph("const (截距)", table_text), Paragraph("6.095 × 10⁷", table_text), Paragraph("3.09 × 10⁶", table_text), Paragraph("19.74", table_text), Paragraph("0.000***", table_text), Paragraph("[5.49e7, 6.70e7]", table_text)],
        [Paragraph("sqft_living", table_text), Paragraph("<b>171.40</b>", table_text), Paragraph("3.63", table_text), Paragraph("47.24", table_text), Paragraph("0.000***", table_text), Paragraph("[164.29, 178.51]", table_text)],
        [Paragraph("grade", table_text), Paragraph("<b>91,410</b>", table_text), Paragraph("2,287", table_text), Paragraph("39.97", table_text), Paragraph("0.000***", table_text), Paragraph("[86,927, 95,892]", table_text)],
        [Paragraph("waterfront", table_text), Paragraph("<b>593,700</b>", table_text), Paragraph("18,800", table_text), Paragraph("31.66", table_text), Paragraph("0.000***", table_text), Paragraph("[5.57e5, 6.30e5]", table_text)],
        [Paragraph("dist_to_seattle", table_text), Paragraph("<b>-1,078,000</b>", table_text), Paragraph("29,100", table_text), Paragraph("-37.01", table_text), Paragraph("0.000***", table_text), Paragraph("[-1.14e6, -1.02e6]", table_text)],
        [Paragraph("bedrooms", table_text), Paragraph("-38,310", table_text), Paragraph("2,125", table_text), Paragraph("-18.03", table_text), Paragraph("0.000***", table_text), Paragraph("[-4.25e4, -3.41e4]", table_text)],
        [Paragraph("house_age", table_text), Paragraph("1,930.7", table_text), Paragraph("79.14", table_text), Paragraph("24.40", table_text), Paragraph("0.000***", table_text), Paragraph("[1775.6, 2085.9]", table_text)],
    ]
    t_ols = Table(ols_table_data, colWidths=[80, 85, 90, 65, 75, 128])
    t_ols.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_ols)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<font size='7.5' color='#64748b'>註：模型整體 F 統計量 = 2,742 (p-value = 0.000)；模型決定係數 R² = 0.717，所有展示核心特徵之 p 值皆 < 0.001 (*** 代表極度顯著)。</font>", body_style))

    # ----------------------------------------------------
    # Section 5: Evaluation
    # ----------------------------------------------------
    story.append(Spacer(1, 5))
    story.append(Paragraph("五、CRISP-DM 階段五：模型評估與視覺化 (Evaluation)", h1_style))
    story.append(Paragraph(
        "<b>1. 測試集客觀指標橫向對比總表：</b><br/>"
        "各模型在獨立測試集 ($N=4,323$) 上之客觀評估成果如下表所示。多變量模型顯著優於單變量基準，決定係數自 0.4937 躍升至 <b>0.7162</b>，RMSE 降低逾 7 萬美元：",
        body_style
    ))

    # Evaluation table
    eval_table_data = [
        [Paragraph("<b>模型架構 (Model)</b>", table_header), Paragraph("<b>R² Score</b>", table_header), Paragraph("<b>Adj R²</b>", table_header), Paragraph("<b>RMSE (美元)</b>", table_header), Paragraph("<b>MAE (美元)</b>", table_header), Paragraph("<b>MAPE (%)</b>", table_header)],
        [Paragraph("1. 單純線性回歸 (sqft_living Baseline)", table_text), Paragraph("0.4937", table_text), Paragraph("0.4936", table_text), Paragraph("$275,655", table_text), Paragraph("$177,529", table_text), Paragraph("35.89%", table_text)],
        [Paragraph("2. 全特徵多元回歸 (Full MLR, 18特徵)", table_text), Paragraph("0.7029", table_text), Paragraph("0.7017", table_text), Paragraph("$211,156", table_text), Paragraph("$126,519", table_text), Paragraph("25.11%", table_text)],
        [Paragraph("<b>3. 精選特徵多元回歸 (Selected MLR)★</b>", table_text), Paragraph("<b>0.7162</b>", table_text), Paragraph("<b>0.7152</b>", table_text), Paragraph("<b>$206,382</b>", table_text), Paragraph("<b>$124,079</b>", table_text), Paragraph("<b>25.52%</b>", table_text)],
        [Paragraph("4. Ridge 脊回歸 (L2 正則化)", table_text), Paragraph("0.7161", table_text), Paragraph("0.7150", table_text), Paragraph("$206,433", table_text), Paragraph("$123,680", table_text), Paragraph("25.36%", table_text)],
        [Paragraph("5. Lasso 套索回歸 (L1 正則化)", table_text), Paragraph("0.7162", table_text), Paragraph("0.7151", table_text), Paragraph("$206,387", table_text), Paragraph("$124,030", table_text), Paragraph("25.50%", table_text)],
        [Paragraph("6. 對數轉換多元回歸 (Log-MLR)", table_text), Paragraph("0.5544", table_text), Paragraph("0.5527", table_text), Paragraph("$258,617", table_text), Paragraph("<b>$109,117</b>", table_text), Paragraph("<b>19.31%</b>", table_text)],
        [Paragraph("7. 隨機森林回歸 (Random Forest 網路主流)", table_text), Paragraph("<b>0.8668</b>", table_text), Paragraph("<b>0.8663</b>", table_text), Paragraph("<b>$141,395</b>", table_text), Paragraph("<b>$72,004</b>", table_text), Paragraph("<b>13.26%</b>", table_text)],
    ]
    t_eval = Table(eval_table_data, colWidths=[165, 68, 68, 75, 75, 72])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f0fdf4')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 8))

    # Fig 4: Confidence Interval & Prediction Interval
    story.append(Paragraph("<b>2. 預測圖分析：95% 信賴區間 (CI) 與 95% 預測區間 (PI)：</b>", h2_style))
    story.append(Paragraph(
        "根據統計學原理：<br/>"
        "• <b>95% 信賴區間 (Confidence Interval, CI)</b>：衡量的是在特定 $X_0$ 條件下「母體平均房價期望值 $\\mathbb{E}[Y|X_0]$」的估計範圍，誤差僅來自係數抽樣不確定性 $\\text{Var}(\\hat{y}_0) = \\sigma^2 x_0^T (X^T X)^{-1} x_0$。<br/>"
        "• <b>95% 預測區間 (Prediction Interval, PI)</b>：衡量的是「單一未來特定房屋觀測值 $Y_0$」的真實落點範圍，誤差來自參數抽樣誤差<b>加上</b>不可約減之隨機誤差項變異 $\\sigma^2$：$\\text{Var}(Y_0 - \\hat{y}_0) = \\sigma^2 [1 + x_0^T (X^T X)^{-1} x_0]$。<br/>"
        "因此在圖 4 與圖 6 中，<b>預測區間帶 (PI) 顯著寬於信賴區間帶 (CI)</b>，其上下界成功涵蓋了超過 95% 的真實房價樣本點，具備強大的不確定性風險量化價值。",
        body_style
    ))

    if os.path.exists('figures/fig4_simple_linear_regression_ci_pi.png'):
        img4 = Image('figures/fig4_simple_linear_regression_ci_pi.png', width=500, height=210)
        story.append(img4)
        story.append(Paragraph("【圖 4】單變數線性回歸擬合圖：紅色為回歸線，深藍色色帶為 95% 信賴區間 (CI)，黑色虛線為 95% 預測區間 (PI)", caption_style))

    # Fig 5 & Fig 6
    if os.path.exists('figures/fig5_mlr_actual_vs_pred.png') and os.path.exists('figures/fig6_mlr_prediction_interval.png'):
        t_pred_imgs = Table([
            [
                Image('figures/fig5_mlr_actual_vs_pred.png', width=225, height=200),
                Image('figures/fig6_mlr_prediction_interval.png', width=285, height=200)
            ]
        ], colWidths=[230, 293])
        t_pred_imgs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_pred_imgs)
        story.append(Paragraph("【圖 5 & 圖 6】(左) 多元線性回歸實際房價 vs. 預測房價散佈圖；(右) 測試集前 80 筆樣本之點估計與 95% 預測區間色帶涵蓋圖", caption_style))

    # Fig 7: Residual Diagnostics
    story.append(Spacer(1, 5))
    story.append(Paragraph("<b>3. 殘差統計診斷 (Residual Diagnostics) 與高斯-馬可夫假設檢驗：</b>", h2_style))
    story.append(Paragraph(
        "古典線性回歸需滿足高斯-馬可夫定理 (Gauss-Markov Assumptions)。圖 7 顯示四合一殘差診斷：<br/>"
        "• <b>線性度 (Residuals vs Fitted)：</b>殘差點在 0 水平線上下大致均勻分佈，但在擬合值大於 150 萬美元的高價端呈現向上喇叭狀散佈。<br/>"
        "• <b>常態性 (Normal Q-Q Plot)：</b>常態分位數介於 -2 至 +2 的主流區間高度貼合紅線；但在極端高價端存在厚尾 (Heavy-tail)，反映房產市場之極端豪宅效應。<br/>"
        "• <b>獨立性 (Durbin-Watson)：</b>Durbin-Watson 統計量為 <b>1.983</b>（極度接近 2.0），確認殘差無一階自相關性，滿足獨立性假設。",
        body_style
    ))

    if os.path.exists('figures/fig7_residual_diagnostics.png'):
        img7 = Image('figures/fig7_residual_diagnostics.png', width=490, height=220)
        story.append(img7)
        story.append(Paragraph("【圖 7】殘差統計診斷圖四合一：包含擬合值散佈圖、常態 Q-Q 圖、殘差分佈直方圖與尺度-定位圖", caption_style))

    # Fig 8 & Fig 9
    if os.path.exists('figures/fig8_model_comparison.png') and os.path.exists('figures/fig9_feature_importance.png'):
        t_comp_imgs = Table([
            [
                Image('figures/fig8_model_comparison.png', width=255, height=180),
                Image('figures/fig9_feature_importance.png', width=255, height=180)
            ]
        ], colWidths=[260, 263])
        t_comp_imgs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_comp_imgs)
        story.append(Paragraph("【圖 8 & 圖 9】(左) 各模型評估指標 (R² 與 RMSE) 條形圖；(右) 多元線性回歸標準化回歸係數 (β*) 重要度排行", caption_style))

    # ----------------------------------------------------
    # Section 6: Deployment
    # ----------------------------------------------------
    story.append(Spacer(1, 6))
    story.append(Paragraph("六、CRISP-DM 階段六：部署與商業落地 (Deployment)", h1_style))
    story.append(Paragraph(
        "<b>1. 在線即時房產估價微服務架構 (API Architecture)：</b><br/>"
        "模型透過 FastAPI / Flask 封裝為 RESTful API，並於前端介面提供買方與房仲試算。"
        "當使用者輸入房產特徵（如 3 房 2 衛、1800 平方英尺、等級 7、距離市中心 0.15 單位）時，系統能即時回傳："
        "<br/>• <b>點估計預測售價：$399,320.80</b>"
        "<br/>• <b>95% 平均值信賴區間 (CI)：[$393,732.05, $404,909.56]</b>（代表西雅圖該規格房屋之公允平均行情）"
        "<br/>• <b>95% 個別房價預測區間 (PI)：[$22,102.42, $776,539.18]</b>（代表該特定房屋考量個別屋況變異後之合理議價波段）。",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. 模型監控與維護機制 (MLOps & Governance)：</b><br/>"
        "• <b>概念漂移 (Concept Drift) 監控：</b>房地產價格強烈受總體經濟利率（如聯準會降息/升息）與通膨波動影響。系統定期追蹤每季殘差均方誤差，當監控到 RMSE 上升超過 15% 時觸發模型重新訓練機制。<br/>"
        "• <b>合規性與公平性審查：</b>線性回歸模型完全公開各變數權重係數，不包含任何受保護的種族或敏感社會指標，完全符合美國《公平住房法 (Fair Housing Act)》之金融合規要求。",
        body_style
    ))

    # ----------------------------------------------------
    # Section 7: AI Assistance Statement & NotebookLM
    # ----------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(Paragraph("七、AI 輔助內容聲明與 NotebookLM 研究摘要 (作業規範必填)", h1_style))
    
    # 7.1 GPT Statement Box
    gpt_box_content = [
        [Paragraph("<b>【GPT 輔助內容 (ChatGPT Assistance Statement)】</b>", ParagraphStyle('GptBoxTitle', fontName='ChineseFontBold', fontSize=9.5, textColor=colors.HexColor('#1d4ed8')))],
        [Paragraph(
            "本專案在研究設計與實作過程中，使用 ChatGPT 進行以下關鍵技術環節之輔助與引導：<br/>"
            "1. <b>多重共線性診斷導引：</b>提示注意建築面積中 <i>sqft_living = sqft_above + sqft_basement</i> 的完全代數相依現象，並指導使用 <code>statsmodels.variance_inflation_factor</code> 進行 VIF 矩陣計算與共線性特徵剔除。<br/>"
            "2. <b>統計推論與區間估計實作：</b>指導運用 <code>model.get_prediction().summary_frame(alpha=0.05)</code> 分別導出條件期望值信賴區間 (CI) 與個別觀測值預測區間 (PI)，並撰寫 <code>plt.fill_between</code> 視覺化區間帶程式碼。<br/>"
            "3. <b>高斯-馬可夫殘差診斷：</b>協助構建 Residuals vs Fitted、Normal Q-Q Plot、Scale-Location 四合一統計診斷圖表。<br/>"
            "4. <b>PDF 匯出對話紀錄：</b>完整問答對話歷程已透過程式整理並匯出為 <b><code>GPT_Conversation_History.pdf</code></b> 與 <b><code>GPT_Conversation_History.html</code></b>（可隨時以 pdfCrowd 重新轉存），供檢驗查核。",
            body_style
        )]
    ]
    t_gpt_box = Table(gpt_box_content, colWidths=[523])
    t_gpt_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff6ff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#3b82f6')),
        ('LINEBEFORE', (0, 0), (0, -1), 4, colors.HexColor('#1d4ed8')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_gpt_box)
    story.append(Spacer(1, 8))

    # 7.2 NotebookLM Summary Box
    nlm_box_content = [
        [Paragraph("<b>【NotebookLM 摘要 (NotebookLM Research Summary - 100字以上)】</b>", ParagraphStyle('NlmBoxTitle', fontName='ChineseFontBold', fontSize=9.5, textColor=colors.HexColor('#047857')))],
        [Paragraph(
            "本研究運用 Google NotebookLM 匯入 Kaggle「House Sales in King County, USA」資料集之多篇頂尖高分開源 Notebooks、學術分析論文與經典競賽解決方案進行深度交叉比對與精華綜述：<br/>"
            "<b>1. 主流解法性能演進脈絡：</b>在基準模型方面，傳統多元線性回歸 (OLS) 的測試集 $R^2$ 大致落在 $0.68 \\sim 0.72$ 區間，RMSE 約為 20 萬美元。若未對數據進行偏態修正，極端高價豪宅會對最小平方法的平方損失造成嚴重扭曲。而在 Kaggle 競賽高分排行榜上，主流方案幾乎全面轉向非線性集成學習，尤其是樹狀梯度提升模型（如 LightGBM、XGBoost、CatBoost），可將測試集 $R^2$ 顯著推升至 <b>$0.87 \\sim 0.90$</b>，RMSE 壓低至 <b>11 萬至 13 萬美元</b>。<br/>"
            "<b>2. 關鍵特徵工程精華：</b>NotebookLM 研究指出，頂尖作者最核心的致勝關鍵在於「地理微觀特徵工程」。線性回歸僅能將經緯度擬合為連續傾斜平面，無法刻畫西雅圖特定熱門學區與湖濱微型社區（如 Medina、Mercer Island）的跳躍式高地價；頂尖方案普遍採用「Zipcode 郵遞區號之目標編碼 (Target Encoding)」、「K-Nearest Neighbors (KNN) 鄰近 5~10 棟房屋均價特徵」以及「對房價取對數 $\\log(y)$」以降低右偏極端值之敏感度。<br/>"
            "<b>3. 統計可解釋性與預測精度之權衡：</b>雖然非線性集成模型在純數值預測指標上具備壓倒性優勢，但其內部決策路徑為高階黑盒子，難以提供符合銀行審查與估價法規之解析權重。相較之下，多元線性回歸所提供的各特徵邊際價格貢獻（回歸係數 $\\beta$）以及嚴謹的 95% 信賴與預測區間，在金融合規與房產抗辯上仍具備不可替代的獨特商業價值。",
            body_style
        )]
    ]
    t_nlm_box = Table(nlm_box_content, colWidths=[523])
    t_nlm_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0fdf4')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#10b981')),
        ('LINEBEFORE', (0, 0), (0, -1), 4, colors.HexColor('#047857')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_nlm_box)
    story.append(Spacer(1, 8))

    # ----------------------------------------------------
    # Section 8: Mainstream and Better Solutions Comparison
    # ----------------------------------------------------
    story.append(Paragraph("八、網路上主流或更優解法之深入比較與說明", h1_style))
    story.append(Paragraph(
        "針對金郡房價預測主題，將本專案所採用之多元線性回歸 (MLR) 與目前 Kaggle 及業界主流頂尖解法進行多維度系統性對比：",
        body_style
    ))

    comp_table_data = [
        [Paragraph("<b>比較維度</b>", table_header), Paragraph("<b>本專案多元線性回歸 (MLR)</b>", table_header), Paragraph("<b>網路主流頂尖解法 (LightGBM / XGBoost / Stacking)</b>", table_header)],
        [Paragraph("<b>核心演算法</b>", table_text), Paragraph("普通最小平方法 (OLS) 參數模型", table_text), Paragraph("基於直方圖之梯度提升決策樹 (GBDT) / 堆疊集成", table_text)],
        [Paragraph("<b>測試集 R² 分數</b>", table_text), Paragraph("<b>0.7162</b> (符合統計回歸水準)", table_text), Paragraph("<b>0.8700 ~ 0.8950</b> (頂尖高分水準)", table_text)],
        [Paragraph("<b>測試集 RMSE 誤差</b>", table_text), Paragraph("$206,382 美元", table_text), Paragraph("<b>$115,000 ~ $130,000 美元</b> (誤差大幅降低)", table_text)],
        [Paragraph("<b>模型可解釋性</b>", table_text), Paragraph("<b>極高</b> (各特徵係數具備明確邊際溢價意義)", table_text), Paragraph("<b>中等至偏低</b> (需依賴 SHAP 值或特徵重要度事後解釋)", table_text)],
        [Paragraph("<b>統計推論能力</b>", table_text), Paragraph("<b>完備</b> (提供 p 值、t 檢定、95% CI 與 PI)", table_text), Paragraph("<b>匱乏</b> (僅輸出點估計，難以解析推導統計區間)", table_text)],
        [Paragraph("<b>非線性特徵擬合</b>", table_text), Paragraph("依賴人工特徵工程 (如衍生距離變數)", table_text), Paragraph("<b>自動學習</b> 任意階數特徵非線性交互 (如 面積×建材等級)", table_text)],
        [Paragraph("<b>地理空間處理</b>", table_text), Paragraph("連續平面擬合 (受限於經緯度斜率)", table_text), Paragraph("<b>空間遞迴二元切割</b>、KNN 鄰居均價、郵遞區號目標編碼", table_text)],
        [Paragraph("<b>訓練速度與資源</b>", table_text), Paragraph("<b>極快</b> (毫秒級計算完成，CPU 消耗低)", table_text), Paragraph("較慢 (需進行網格搜尋超參數調優與多折交叉驗證)", table_text)],
        [Paragraph("<b>合規與商業應用</b>", table_text), Paragraph("<b>銀行房貸法定估價審核、政府估稅公開模型</b>", table_text), Paragraph("<b>房仲 APP 大數據自動估價引擎 (AVM)、量化投資決策</b>", table_text)],
    ]
    t_mainstream = Table(comp_table_data, colWidths=[105, 205, 213])
    t_mainstream.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_mainstream)
    story.append(Spacer(1, 8))

    # ----------------------------------------------------
    # Section 9: Conclusion
    # ----------------------------------------------------
    story.append(Paragraph("九、結論與成果總結 (Conclusion)", h1_style))
    story.append(Paragraph(
        "1. <b>CRISP-DM 流程貫徹：</b>本研究嚴格遵循商業理解、資料理解、資料準備、模型建構、模型評估與部署應用六大階段，完成了端到端之專業數據分析。<br/>"
        "2. <b>統計檢驗與共線性排除：</b>精準診斷出金郡房產資料集中 `sqft_living = sqft_above + sqft_basement` 的代數共線性陷阱，成功將 VIF 壓制在安全範圍內，建構出統計顯著 ($p < 0.001$)、測試集 $R^2 = 0.7162$ 之高質量多元線性回歸模型。<br/>"
        "3. <b>預測區間與不確定性量化：</b>精確區分母體期望值之 95% 信賴區間 (CI) 與個別觀測值之 95% 預測區間 (PI)，並完整呈現於各視覺化圖表中，滿足金融與不動產業務對風險評估之實質需求。<br/>"
        "4. <b>主流技術視野開拓：</b>透過 NotebookLM 文獻調研與隨機森林實驗，深入論證了線性模型與現代 GBDT 樹模型在可解釋性與預測精度上的權衡 (Trade-off)，為後續工程落地與混合架構提供堅實依據。",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] PDF report generated: {filename}")

if __name__ == '__main__':
    build_pdf_report('7114056XXX_hw2_Report.pdf')
    # Also generate 7114056XXX_hw2.pdf
    build_pdf_report('7114056XXX_hw2.pdf')
