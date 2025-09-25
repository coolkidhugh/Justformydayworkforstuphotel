import re
import streamlit as st
from PIL import Image
import pandas as pd
import io
import json
import unicodedata
import os
# [关键更新] 导入新的侧边栏组件
from streamlit_option_menu import option_menu

# --- SDK 依赖 ---
# requirements.txt needs to include: alibabacloud_ocr_api20210707, pandas, streamlit, pillow, openpyxl, streamlit-option-menu
try:
    from alibabacloud_ocr_api20210707.client import Client as OcrClient
    from alibabacloud_tea_openapi import models as open_api_models
    from alibabacloud_ocr_api20210707 import models as ocr_models
    ALIYUN_SDK_AVAILABLE = True
except ImportError:
    ALIYUN_SDK_AVAILABLE = False


# --- MOCK FUNCTION for Report Analyzer ---
def analyze_reports_ultimate(file_paths):
    """
    一个模拟函数，用于替代缺失的 analyze_excel.py 模块。
    它会生成一些示例分析结果。
    """
    summaries = []
    unknown_codes = {"XYZ": 2, "ABC": 5} 

    for path in file_paths:
        file_name = os.path.basename(path)
        summary = f"### 文件 '{file_name}' 的分析摘要:\n"
        summary += f"- **总房间数**: {len(path) * 5 % 50 + 10}\n"
        summary += f"- **总人数**: {len(path) * 8 % 80 + 20}\n"
        summary += "- **关键发现**: 此报告中的 VIP 客人数量较多。\n---"
        summaries.append(summary)

    return summaries, unknown_codes


# ==============================================================================
# --- APP 1: OCR 销售通知生成器 ---
# ==============================================================================
def run_ocr_app():
    """Contains all logic and UI for the OCR Sales Notification Generator."""

    # --- 配置信息 ---
    TEAM_TYPE_MAP = { "CON": "会议团", "FIT": "散客团", "WA": "婚宴团" }
    DEFAULT_TEAM_TYPE = "旅游团"
    ALL_ROOM_CODES = [
        "DETN", "DKN", "DQN", "DSKN", "DSTN", "DTN", "EKN", "EKS", "ESN", "ESS",
        "ETN", "ETS", "FSN", "FSB", "FSC", "OTN", "PSA", "PSB", "RSN", "SKN",
        "SQN", "SQS", "SSN", "SSS", "STN", "STS", "JDEN", "JDKN", "JDKS", "JEKN",
        "JESN", "JESS", "JETN", "JETS", "JKN", "JLKN", "JTN", "JTS", "PSC", "PSD",
        "VCKN", "VCKD", "SITN", "JEN", "JIS", "JTIN"
    ]

    # --- 登录检查函数 ---
    def check_password():
        """返回 True 如果用户已登录, 否则返回 False."""
        def login_form():
            with st.form("Credentials"):
                st.text_input("用户名", key="username")
                st.text_input("密码", type="password", key="password")
                st.form_submit_button("登录", on_click=password_entered)

        def password_entered():
            app_username = os.environ.get("APP_USERNAME")
            app_password = os.environ.get("APP_PASSWORD")
            
            if st.session_state["username"] == app_username and st.session_state["password"] == app_password:
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False

        if not os.environ.get("APP_USERNAME") or not os.environ.get("APP_PASSWORD"):
            st.error("错误：应用的用户名和密码未在 Render 的环境变量中正确配置。请参考指南操作。")
            return False

        if st.session_state.get("password_correct", False):
            return True

        login_form()
        if "password_correct" in st.session_state and not st.session_state.password_correct:
            st.error("😕 用户名或密码不正确。")
        return False

    # --- OCR 引擎函数 (阿里云版) ---
    def get_ocr_text_from_aliyun(image: Image.Image) -> str:
        if not ALIYUN_SDK_AVAILABLE:
            st.error("错误：阿里云 SDK 未安装。请确保 requirements.txt 文件配置正确。")
            return None
        
        access_key_id = os.environ.get("ALIYUN_ACCESS_KEY_ID")
        access_key_secret = os.environ.get("ALIYUN_ACCESS_KEY_SECRET")

        if not access_key_id or not access_key_secret:
            st.error("错误：阿里云凭证未在 Render 的环境变量中配置。")
            return None
            
        try:
            config = open_api_models.Config(
                access_key_id=access_key_id,
                access_key_secret=access_key_secret,
                endpoint='ocr-api.cn-hangzhou.aliyuncs.com'
            )
            client = OcrClient(config)

            buffered = io.BytesIO()
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            image_format = "JPEG"
            image.save(buffered, format=image_format)
            buffered.seek(0)
            
            request = ocr_models.RecognizeGeneralRequest(body=buffered)
            response = client.recognize_general(request)

            if response.status_code == 200 and response.body and response.body.data:
                data = json.loads(response.body.data)
                return data.get('content', '')
            else:
                error_message = '无详细信息'
                if response.body and hasattr(response.body, 'message'):
                   error_message = response.body.message
                raise Exception(f"阿里云 OCR API 返回错误: {error_message}")

        except Exception as e:
            st.error(f"调用阿里云 OCR API 失败: {e}")
            return None

    # --- 信息提取与格式化 ---
    def extract_booking_info(ocr_text: str):
        team_name_pattern = re.compile(r'((?:CON|FIT|WA)\d+\s*/\s*[\u4e00-\u9fa5\w]+)', re.IGNORECASE)
        date_pattern = re.compile(r'(\d{1,2}/\d{1,2})')

        team_name_match = team_name_pattern.search(ocr_text)
        if not team_name_match: return "错误：无法识别出团队名称。"
        team_name = re.sub(r'\s*/\s*', '/', team_name_match.group(1).strip())

        all_dates = date_pattern.findall(ocr_text)
        unique_dates = sorted(list(set(all_dates)))
        if not unique_dates: return "错误：无法识别出有效的日期。"
        arrival_date = unique_dates[0]
        departure_date = unique_dates[-1]

        room_codes_pattern_str = '|'.join(ALL_ROOM_CODES)
        room_finder_pattern = re.compile(f'({room_codes_pattern_str})\\s*(\\d+)', re.IGNORECASE)
        price_finder_pattern = re.compile(r'\b(\d+\.\d{2})\b')

        found_rooms = [(m.group(1).upper(), int(m.group(2)), m.span()) for m in room_finder_pattern.finditer(ocr_text)]
        found_prices = [(float(m.group(1)), m.span()) for m in price_finder_pattern.finditer(ocr_text)]

        room_details = []
        available_prices = list(found_prices)

        for room_type, num_rooms, room_span in found_rooms:
            best_price = None
            best_price_index = -1
            min_distance = float('inf')

            for i, (price_val, price_span) in enumerate(available_prices):
                if price_span[0] > room_span[1]:
                    distance = price_span[0] - room_span[1]
                    if distance < min_distance:
                        min_distance = distance
                        best_price = price_val
                        best_price_index = i
            
            if best_price is not None and best_price > 0:
                room_details.append((room_type, num_rooms, int(best_price)))
                if best_price_index != -1:
                    available_prices.pop(best_price_index)

        if not room_details:
            return f"提示：找到了团队 {team_name}，但未能自动匹配任何有效的房型和价格。请检查原始文本并手动填写。"

        team_prefix = team_name[:3].upper()
        team_type = TEAM_TYPE_MAP.get(team_prefix, DEFAULT_TEAM_TYPE)
        room_details.sort(key=lambda x: x[1])

        try:
            arr_month, arr_day = map(int, arrival_date.split('/'))
            dep_month, dep_day = map(int, departure_date.split('/'))
            formatted_arrival = f"{arr_month}月{arr_day}日"
            formatted_departure = f"{dep_month}月{dep_day}日"
        except (ValueError, IndexError):
            return "错误：日期格式无法解析。"

        df = pd.DataFrame(room_details, columns=['房型', '房数', '定价'])
        return {"team_name": team_name, "team_type": team_type, "arrival_date": formatted_arrival, "departure_date": formatted_departure, "room_dataframe": df}

    def format_notification_speech(team_name, team_type, arrival_date, departure_date, room_df):
        date_range_string = f"{arrival_date}至{departure_date}"
        room_details = room_df.to_dict('records')
        formatted_rooms = [f"{item['房数']}间{item['房型']}({item['定价']})" for item in room_details]
        room_string = " ".join(formatted_rooms) if formatted_rooms else "无房间详情"
        return f"新增{team_type} {team_name} {date_range_string} {room_string}。销售通知"

    # --- Streamlit 主应用 ---
    st.title("金陵富士康一秒boom - OCR 工具")

    if check_password():
        st.markdown("""
        **全新工作流**：
        1.  **上传图片，点击提取**：程序将调用阿里云 OCR 并将**原始识别文本**显示在下方。
        2.  **自动填充与人工修正**：程序会尝试自动填充结构化信息。您可以**参照原始文本**，直接在表格中修改，确保信息完全准确。
        3.  **生成话术**：确认无误后，生成最终话术。
        """)

        uploaded_file = st.file_uploader("上传图片文件", type=["png", "jpg", "jpeg", "bmp"], key="ocr_uploader")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", width=300)

            if st.button("从图片提取信息 (阿里云 OCR)"):
                for key in ['raw_ocr_text', 'booking_info']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                with st.spinner('正在调用阿里云 OCR API 识别中...'):
                    ocr_text = get_ocr_text_from_aliyun(image)
                    if ocr_text:
                        st.session_state['raw_ocr_text'] = ocr_text
                        result = extract_booking_info(ocr_text)
                        if isinstance(result, str):
                            st.warning(f"自动解析提示：{result}")
                            st.info("请参考下方识别出的原始文本，手动填写信息。")
                            empty_df = pd.DataFrame(columns=['房型', '房数', '定价'])
                            st.session_state['booking_info'] = { "team_name": "", "team_type": DEFAULT_TEAM_TYPE, "arrival_date": "", "departure_date": "", "room_dataframe": empty_df }
                        else:
                            st.session_state['booking_info'] = result
                            st.success("信息提取成功！请在下方核对并编辑。")

        if 'booking_info' in st.session_state:
            info = st.session_state['booking_info']
            if 'raw_ocr_text' in st.session_state:
                st.markdown("---")
                st.subheader("原始识别结果 (供参考)")
                st.text_area("您可以从这里复制内容来修正下面的表格", st.session_state['raw_ocr_text'], height=200)
            st.markdown("---")
            st.subheader("核对与编辑信息")
            col1, col2, col3, col4 = st.columns(4)
            with col1: info['team_name'] = st.text_input("团队名称", value=info['team_name'])
            with col2: info['team_type'] = st.selectbox("团队类型", options=list(TEAM_TYPE_MAP.values()) + [DEFAULT_TEAM_TYPE], index=(list(TEAM_TYPE_MAP.values()) + [DEFAULT_TEAM_TYPE]).index(info['team_type']))
            with col3: arrival = st.text_input("到达日期", value=info['arrival_date'])
            with col4: departure = st.text_input("离开日期", value=info['departure_date'])
            st.markdown("##### 房间详情 (可直接在表格中编辑)")
            edited_df = st.data_editor(info['room_dataframe'], num_rows="dynamic", use_container_width=True)
            if st.button("生成最终话术"):
                final_speech = format_notification_speech(info['team_name'], info['team_type'], arrival, departure, edited_df)
                st.subheader("生成成功！")
                st.success(final_speech)
                st.code(final_speech, language=None)

# ==============================================================================
# --- APP 2: 多维审核比对平台 ---
# ==============================================================================
def run_comparison_app():
    """Contains all logic and UI for the Data Comparison Platform."""
    SESSION_DEFAULTS = {
        'df1': None, 'df2': None, 'df1_name': "", 'df2_name': "",
        'ran_comparison': False, 'common_rows': pd.DataFrame(),
        'matched_df': pd.DataFrame(), 'in_file1_only': pd.DataFrame(),
        'in_file2_only': pd.DataFrame(), 'compare_cols_keys': []
    }
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    def forensic_clean_text(text):
        if not isinstance(text, str): return text
        try:
            cleaned_text = unicodedata.normalize('NFKC', text)
        except (TypeError, ValueError):
            return text
        cleaned_text = re.sub(r'[\u200B-\u200D\uFEFF\s\xa0]+', '', cleaned_text)
        return cleaned_text.strip()

    def process_and_standardize(df, mapping, case_insensitive=False, room_type_equivalents=None):
        if not mapping.get('name'):
            return pd.DataFrame()

        standard_df = pd.DataFrame()
        for col_key, col_name in mapping.items():
            if col_name and col_name in df.columns:
                standard_df[col_key] = df[col_name]

        def robust_date_parser(series):
            def process_date(date_str):
                if pd.isna(date_str): return pd.NaT
                date_str = str(date_str).strip()
                if re.match(r'^\d{1,2}/\d{1,2}', date_str):
                    date_part = date_str.split(' ')[0]
                    return f"2025-{date_part.replace('/', '-')}"
                return date_str
            return pd.to_datetime(series.apply(process_date), errors='coerce').dt.strftime('%Y-%m-%d')

        if 'start_date' in standard_df.columns:
            standard_df['start_date'] = robust_date_parser(standard_df['start_date'])
        if 'end_date' in standard_df.columns:
            standard_df['end_date'] = robust_date_parser(standard_df['end_date'])

        if 'room_type' in standard_df.columns:
            standard_df['room_type'] = standard_df['room_type'].astype(str).apply(forensic_clean_text)
            if room_type_equivalents:
                cleaned_equivalents = {forensic_clean_text(k): [forensic_clean_text(val) for val in v] for k, v in room_type_equivalents.items()}
                reverse_map = {val: key for key, values in cleaned_equivalents.items() for val in values}
                standard_df['room_type'] = standard_df['room_type'].replace(reverse_map)

        if 'price' in standard_df.columns:
            standard_df['price'] = pd.to_numeric(standard_df['price'].astype(str).str.strip(), errors='coerce')

        standard_df['name'] = standard_df['name'].astype(str).str.split(r'[、,，/]')
        standard_df = standard_df.explode('name')
        standard_df['name'] = standard_df['name'].apply(forensic_clean_text)

        if case_insensitive:
            standard_df['name'] = standard_df['name'].str.lower()

        standard_df = standard_df[standard_df['name'] != ''].dropna(subset=['name']).reset_index(drop=True)
        return standard_df

    def highlight_diff(row, col1, col2):
        style = 'background-color: #FFC7CE'
        if row.get(col1) != row.get(col2) and not (pd.isna(row.get(col1)) and pd.isna(row.get(col2))):
            return [style] * len(row)
        return [''] * len(row)

    st.title("金陵富士康一秒boom - 比对平台")
    st.info("全新模式：结果以独立的标签页展示，并内置智能日期统一引擎，比对更精准！")

    st.header("第 1 步: 上传文件")
    if st.button("清空并重置"):
        for key in SESSION_DEFAULTS.keys():
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file1 = st.file_uploader("上传名单文件 1", type=['csv', 'xlsx'], key="comp_uploader1")
        if uploaded_file1:
            st.session_state.df1 = pd.read_excel(uploaded_file1) if uploaded_file1.name.endswith('xlsx') else pd.read_csv(uploaded_file1)
            st.session_state.df1_name = uploaded_file1.name
    with col2:
        uploaded_file2 = st.file_uploader("上传名单文件 2", type=['csv', 'xlsx'], key="comp_uploader2")
        if uploaded_file2:
            st.session_state.df2 = pd.read_excel(uploaded_file2) if uploaded_file2.name.endswith('xlsx') else pd.read_csv(uploaded_file2)
            st.session_state.df2_name = uploaded_file2.name

    if st.session_state.df1 is not None and st.session_state.df2 is not None:
        st.header("第 2 步: 选择要比对的列 (姓名必选)")
        mapping = {'file1': {}, 'file2': {}}
        cols_to_map = ['name', 'start_date', 'end_date', 'room_type', 'price']
        col_names_zh = ['姓名', '入住日期', '离开日期', '房型', '房价']

        cols1, cols2 = st.columns(2)
        with cols1:
            st.subheader(f"文件 1: {st.session_state.df1_name}")
            df1_cols = [None] + list(st.session_state.df1.columns)
            for key, name_zh in zip(cols_to_map, col_names_zh):
                mapping['file1'][key] = st.selectbox(f"{name_zh}", df1_cols, key=f'f1_{key}')
        with cols2:
            st.subheader(f"文件 2: {st.session_state.df2_name}")
            df2_cols = [None] + list(st.session_state.df2.columns)
            for key, name_zh in zip(cols_to_map, col_names_zh):
                mapping['file2'][key] = st.selectbox(f"{name_zh}", df2_cols, key=f'f2_{key}')

        st.header("第 3 步: 配置与执行")
        room_type_equivalents = {}
        if mapping['file1'].get('room_type') and mapping['file2'].get('room_type'):
            with st.expander("高级功能：统一不同名称的房型 (例如：让'大床房'='King Room')"):
                unique_rooms1 = st.session_state.df1[mapping['file1']['room_type']].dropna().astype(str).unique()
                unique_rooms2 = list(st.session_state.df2[mapping['file2']['room_type']].dropna().astype(str).unique())
                for room1 in unique_rooms1:
                    room_type_equivalents[room1] = st.multiselect(f"文件1的“{room1}”等同于:", unique_rooms2, key=f"map_{room1}")

        case_insensitive = st.checkbox("比对姓名时忽略大小写/全半角", True)

        if st.button("开始比对", type="primary"):
            if not mapping['file1'].get('name') or not mapping['file2'].get('name'):
                st.error("请确保两边文件的“姓名”都已正确选择。")
            else:
                with st.spinner('正在执行终极比对...'):
                    st.session_state.ran_comparison = True
                    st.session_state.df1.sort_values(by=mapping['file1']['name'], inplace=True, ignore_index=True)
                    st.session_state.df2.sort_values(by=mapping['file2']['name'], inplace=True, ignore_index=True)
                    std_df1 = process_and_standardize(st.session_state.df1, mapping['file1'], case_insensitive, room_type_equivalents)
                    std_df2 = process_and_standardize(st.session_state.df2, mapping['file2'], case_insensitive)
                    merged_df = pd.merge(std_df1, std_df2, on='name', how='outer', suffixes=('_1', '_2'))
                    cols1_for_check = [f"{c}_1" for c in std_df1.columns if c != 'name']
                    cols2_for_check = [f"{c}_2" for c in std_df2.columns if c != 'name']
                    both_exist_mask = merged_df[cols1_for_check].notna().any(axis=1) & merged_df[cols2_for_check].notna().any(axis=1)
                    st.session_state.common_rows = merged_df[both_exist_mask].copy().reset_index(drop=True)
                    only_in_1_mask = merged_df[cols1_for_check].notna().any(axis=1) & merged_df[cols2_for_check].isna().all(axis=1)
                    st.session_state.in_file1_only = merged_df[only_in_1_mask].reset_index(drop=True)
                    only_in_2_mask = merged_df[cols1_for_check].isna().all(axis=1) & merged_df[cols2_for_check].notna().any(axis=1)
                    st.session_state.in_file2_only = merged_df[only_in_2_mask].reset_index(drop=True)
                    st.session_state.compare_cols_keys = [key for key in ['start_date', 'end_date', 'room_type', 'price'] if mapping['file1'].get(key) and mapping['file2'].get(key)]
                    if not st.session_state.common_rows.empty and st.session_state.compare_cols_keys:
                        condition = pd.Series(True, index=st.session_state.common_rows.index)
                        for key in st.session_state.compare_cols_keys:
                            condition &= (st.session_state.common_rows[f'{key}_1'] == st.session_state.common_rows[f'{key}_2']) | \
                                         (st.session_state.common_rows[f'{key}_1'].isna() & st.session_state.common_rows[f'{key}_2'].isna())
                        st.session_state.matched_df = st.session_state.common_rows[condition]
                    else:
                        st.session_state.matched_df = st.session_state.common_rows

        if st.session_state.ran_comparison:
            st.header("第 4 步: 查看比对结果")
            tab_list = ["结果总览"]
            tab_name_map = {'start_date': "入住日期", 'end_date': "离开日期", 'room_type': "房型", 'price': "房价"}
            for key in st.session_state.compare_cols_keys:
                tab_list.append(tab_name_map[key])
            tabs = st.tabs(tab_list)

            with tabs[0]:
                st.subheader("宏观统计")
                stat_cols = st.columns(3)
                matched_count = len(st.session_state.matched_df)
                only_1_count = len(st.session_state.in_file1_only)
                only_2_count = len(st.session_state.in_file2_only)
                stat_cols[0].metric("信息完全一致", matched_count)
                stat_cols[1].metric(f"仅 '{st.session_state.df1_name}' 有", only_1_count)
                stat_cols[2].metric(f"仅 '{st.session_state.df2_name}' 有", only_2_count)

                st.subheader("人员名单详情")
                with st.expander(f"查看 {matched_count} 条信息完全一致的名单"):
                    if not st.session_state.matched_df.empty:
                        st.dataframe(st.session_state.matched_df[['name']].rename(columns={'name': '姓名'}))
                    else:
                        st.write("没有信息完全一致的人员。")

                with st.expander(f"查看 {only_1_count} 条仅存在于 '{st.session_state.df1_name}' 的名单"):
                    if not st.session_state.in_file1_only.empty:
                        display_cols_1 = ['name'] + [c for c in cols_to_map if f"{c}_1" in st.session_state.in_file1_only.columns]
                        display_df_1 = st.session_state.in_file1_only[[f"{c}_1" if c != 'name' else 'name' for c in display_cols_1]]
                        display_df_1.columns = [col_names_zh[cols_to_map.index(c)] if c != 'name' else '姓名' for c in display_cols_1]
                        st.dataframe(display_df_1)
                    else:
                        st.write("没有人员。")

                with st.expander(f"查看 {only_2_count} 条仅存在于 '{st.session_state.df2_name}' 的名单"):
                    if not st.session_state.in_file2_only.empty:
                        display_cols_2 = ['name'] + [c for c in cols_to_map if f"{c}_2" in st.session_state.in_file2_only.columns]
                        display_df_2 = st.session_state.in_file2_only[[f"{c}_2" if c != 'name' else 'name' for c in display_cols_2]]
                        display_df_2.columns = [col_names_zh[cols_to_map.index(c)] if c != 'name' else '姓名' for c in display_cols_2]
                        st.dataframe(display_df_2)
                    else:
                        st.write("没有人员。")

            for i, key in enumerate(st.session_state.compare_cols_keys):
                with tabs[i+1]:
                    col1_name, col2_name = f'{key}_1', f'{key}_2'
                    display_name = col_names_zh[cols_to_map.index(key)]
                    st.subheader(f"【{display_name}】比对详情")
                    if not st.session_state.common_rows.empty:
                        compare_df = st.session_state.common_rows[['name', col1_name, col2_name]].copy()
                        compare_df.rename(columns={'name': '姓名', col1_name: f'文件1 - {display_name}', col2_name: f'文件2 - {display_name}'}, inplace=True)
                        styled_df = compare_df.style.apply(highlight_diff, col1=f'文件1 - {display_name}', col2=f'文件2 - {display_name}', axis=1)
                        st.dataframe(styled_df)
                    else:
                        st.info("两个文件中没有共同的人员可供进行细节比对。")

        st.divider()
        st.header("原始数据预览 (点击比对后会按姓名排序)")
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"文件 1: {st.session_state.df1_name}")
            st.dataframe(st.session_state.df1)
        with c2:
            st.caption(f"文件 2: {st.session_state.df2_name}")
            st.dataframe(st.session_state.df2)

# ==============================================================================
# --- APP 3: Excel 报告分析器 ---
# ==============================================================================
def run_analyzer_app():
    """Contains all logic and UI for the Excel Report Analyzer."""
    st.title("金陵富士康一秒boom - 报告分析器")
    st.markdown("---伯爵酒店团队报表分析工具---")

    uploaded_files = st.file_uploader("请上传您的 Excel 报告文件 (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="analyzer_uploader")

    if uploaded_files:
        temp_dir = "./temp_uploaded_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        file_paths = []
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(temp_file_path)

        desired_order = ["次日到达", "次日在住", "次日离店", "后天到达"]
        def sort_key(file_path):
            file_name = os.path.basename(file_path)
            for i, keyword in enumerate(desired_order):
                if keyword in file_name:
                    return i
            return len(desired_order)
        file_paths.sort(key=sort_key)

        if st.button("开始分析"):
            with st.spinner("正在分析中，请稍候..."):
                st.subheader("分析结果")
                summaries, unknown_codes = analyze_reports_ultimate(file_paths)
            
            for summary in summaries:
                st.markdown(summary)

            if unknown_codes:
                st.subheader("侦测到的未知房型代码 (请检查是否需要更新规则)")
                st.info("这是模拟数据，用以展示功能。")
                for code, count in unknown_codes.items():
                    st.write(f"代码: '{code}' (出现了 {count} 次)")
            
            for f_path in file_paths:
                try:
                    os.remove(f_path)
                except OSError as e:
                    st.warning(f"Error removing temp file {f_path}: {e}")
            try:
                os.rmdir(temp_dir)
            except OSError as e:
                st.warning(f"Error removing temp directory {temp_dir}: {e}")
    else:
        st.info("请上传一个或多个 Excel 文件以开始分析。")

    st.markdown("""
    ---
    #### 使用说明：
    1. 点击 "Browse files" 上传您的 Excel 报告。可以同时上传多个文件。
    2. 文件上传后，点击 "开始分析" 按钮。
    3. 分析结果将显示在下方。
    """)


# ==============================================================================
# --- Main App Router ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="金陵富士康一秒boom")

# [关键更新] 使用新的 option_menu 组件
with st.sidebar:
    app_choice = option_menu(
        menu_title="金陵富士康一秒boom",  # required
        options=["OCR 工具", "比对平台", "报告分析器"],  # required
        icons=["camera-reels", "columns-gap", "file-earmark-bar-graph"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

st.sidebar.markdown("---")
st.sidebar.info("这是一个将多个工具集成到一起的应用。")

if app_choice == "OCR 工具":
    run_ocr_app()
elif app_choice == "比对平台":
    run_comparison_app()
elif app_choice == "报告分析器":
    run_analyzer_app()

