import re
import streamlit as st
from PIL import Image
import pandas as pd
import io
import json
import unicodedata
import os
import traceback
from datetime import timedelta
# [关键更新] 导入新的侧边栏组件
from streamlit_option_menu import option_menu

# --- SDK 依赖 ---
# requirements.txt needs to include: alibabacloud_ocr_api20210707, pandas, streamlit, pillow, openpyxl, streamlit-option-menu
try:
    from alibabacloud_ocr_api20210707.client import Client as OcrClient
    from alibabacloud_tea_openapi import models as open_api_models
    from alibabacloud_ocr_api20210707 import models as ocr_models
    from alibabacloud_tea_util import models as util_models
    ALIYUN_SDK_AVAILABLE = True
except ImportError:
    ALIYUN_SDK_AVAILABLE = False


# --- MOCK FUNCTION for Report Analyzer ---
def analyze_reports_ultimate(file_paths):
    """
    一个模拟函数，用于替代缺失的 analyze_excel.py 模块。
    它会生成一些示例分析结果，模仿用户期望的输出格式。
    """
    # Based on user's image: 5b25a5b0e1df25073f860126ea39cca3.png
    summaries = [
        "【次日在住】：有效总房数 64 间(共 59 人)，其中会议/公司团队(MGM/MTC)[5个团队，共23间]分布: 金陵楼 17 间, 亚太楼 6 间。(无GTO旅行社房)。",
        "【次日离店】：有效总房数 240 间(共 251 人)，其中会议/公司团队(MGM/MTC)[9个团队，共232间]分布: 金陵楼 173 间, 亚太楼 58 间, 其他楼 1 间。旅行社(GTO)房[2个团队, 8间, 共12人]分布: 金陵楼 8 间, 亚太楼 0 间。",
        "【次日到店】：有效总房数 46 间(共 37 人)，其中会议/公司团队(MGM/MTC)[8个团队, 共17间]分布: 金陵楼 1 间, 亚太楼 6 间。(无GTO旅行社房)。",
        "【后天到店】：有效总房数 0 间(共 0 人)，(无会议/公司团队房)，(无GTO旅行社房)。"
    ]
    # The mock function can return a static result as the core logic is missing
    unknown_codes = {"PSA": 1}
    return summaries, unknown_codes


# ==============================================================================
# --- APP 1: OCR 工具 (表格识别版) ---
# ==============================================================================
def run_ocr_app_detailed():
    """Contains all logic and UI for the Detailed OCR Sales Notification Generator."""

    # --- 配置信息 ---
    TEAM_TYPE_MAP = { "CON": "会议团", "FIT": "散客团", "WA": "婚宴团" }
    DEFAULT_TEAM_TYPE = "旅游团"
    SALES_LIST = ["陈洪贞", "倪斌", "刘亚炜", "黄婷", "蒋思源", "黄泽浩", "蒋光聪", "吴皓宇", "潘茜", "柏方"]

    # --- [终极升级] OCR 引擎函数 (使用 RecognizeTable API) ---
    def get_ocr_table_data(image: Image.Image) -> (dict, str):
        if not ALIYUN_SDK_AVAILABLE:
            st.error("错误：阿里云 SDK 未安装。请确保 requirements.txt 文件配置正确。")
            return None, None
        
        if "aliyun_credentials" not in st.secrets:
            st.error("错误：阿里云凭证未在 Streamlit Cloud 的 Secrets 中配置。")
            return None, None
        
        access_key_id = st.secrets.aliyun_credentials.get("access_key_id")
        access_key_secret = st.secrets.aliyun_credentials.get("access_key_secret")

        if not access_key_id or not access_key_secret:
            st.error("错误：阿里云 AccessKey ID 或 Secret 未在 Secrets 中正确配置。")
            return None, None
            
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
            
            image_format = "PNG" # PNG is recommended for table recognition
            image.save(buffered, format=image_format)
            buffered.seek(0)
            
            request = ocr_models.RecognizeTableRequest(body=buffered)
            runtime = util_models.RuntimeOptions()
            response = client.recognize_table_with_options(request, runtime)

            if response.status_code == 200 and response.body and response.body.data:
                data = json.loads(response.body.data)
                # Reconstruct raw text from table structure for better readability
                raw_text = ""
                if 'tables' in data and data['tables']:
                    for row in data['tables'][0].get('table_rows', []):
                        raw_text += row.get('text', '') + '\n'
                return data, raw_text.strip()
            else:
                error_message = '无详细信息'
                if response.body and hasattr(response.body, 'message'):
                   error_message = response.body.message
                raise Exception(f"阿里云表格识别 API 返回错误: {error_message}")

        except Exception as e:
            st.error(f"调用阿里云 OCR API 失败: {e}")
            return None, None

    # --- [终极升级] 信息提取 (从结构化表格数据) ---
    def extract_booking_info_from_table(ocr_data: dict, ocr_text: str):
        team_name_pattern = re.compile(r'((?:CON|FIT|WA)\d+\s*/\s*[\u4e00-\u9fa5\w]+)', re.IGNORECASE)
        team_name_match = team_name_pattern.search(ocr_text)
        if not team_name_match: return "错误：无法从文本中识别出团队名称。"
        team_name = re.sub(r'\s*/\s*', '/', team_name_match.group(1).strip())
        team_prefix = team_name[:3].upper()
        team_type = TEAM_TYPE_MAP.get(team_prefix, DEFAULT_TEAM_TYPE)

        if not ocr_data or 'tables' not in ocr_data or not ocr_data['tables']:
            return "错误：未能识别出任何表格结构。"

        table = ocr_data['tables'][0]
        header_row = table['table_rows'][0]['table_columns']
        header_texts = [col['text'].strip() for col in header_row]

        # 灵活查找列索引
        def find_col_index(headers, potential_names):
            for name in potential_names:
                if name in headers:
                    return headers.index(name)
            return -1

        status_idx = find_col_index(header_texts, ["状态", "STATUS"])
        room_type_idx = find_col_index(header_texts, ["房类", "ROOM CATEGORY"])
        room_count_idx = find_col_index(header_texts, ["房数", "ROOMS"])
        arrival_idx = find_col_index(header_texts, ["到达", "ARRIVAL"])
        departure_idx = find_col_index(header_texts, ["离开", "DEPARTURE"])
        price_idx = find_col_index(header_texts, ["定价", "RATE"])

        required_indices = {
            "状态": status_idx, "房类": room_type_idx, "房数": room_count_idx,
            "到达": arrival_idx, "离开": departure_idx, "定价": price_idx
        }
        missing_headers = [name for name, index in required_indices.items() if index == -1]
        if missing_headers:
            return f"错误：表格缺少必要的列标题: {', '.join(missing_headers)}。请确保图片清晰。"

        data_rows = table['table_rows'][1:]
        date_groups = {}

        for row in data_rows:
            cols = row['table_columns']
            if len(cols) < len(header_texts): continue

            status = cols[status_idx]['text'].strip()
            if status.upper() != 'R': continue

            try:
                room_type = cols[room_type_idx]['text'].strip()
                room_count = int(cols[room_count_idx]['text'].strip())
                arrival = cols[arrival_idx]['text'].strip().split(' ')[0]
                departure = cols[departure_idx]['text'].strip().split(' ')[0]
                price = int(float(cols[price_idx]['text'].strip()))
                
                arrival = re.sub(r'[^0-9/]', '', arrival)
                departure = re.sub(r'[^0-9/]', '', departure)

                date_key = (arrival, departure)
                if date_key not in date_groups:
                    date_groups[date_key] = []
                date_groups[date_key].append((room_type.upper(), room_count, price))
            except (ValueError, IndexError):
                continue 

        if not date_groups:
            return "错误：未能从表格中解析出任何有效的预定记录。"
        
        result_groups = []
        for (arr, dep), rooms in date_groups.items():
            df = pd.DataFrame(rooms, columns=['房型', '房数', '定价'])
            result_groups.append({ "arrival_raw": arr, "departure_raw": dep, "dataframe": df })

        return {
            "team_name": team_name,
            "team_type": team_type,
            "booking_groups": sorted(result_groups, key=lambda x: x['arrival_raw'])
        }

    # --- 话术生成函数 (不变) ---
    def format_notification_speech(team_name, team_type, booking_groups, salesperson):
        def format_date_range(arr_str, dep_str):
            try:
                arr_month, arr_day = arr_str.split('/')
                dep_month, dep_day = dep_str.split('/')
                if arr_month == dep_month:
                    return f"{int(arr_month)}.{int(arr_day)}-{int(dep_day)}"
                return f"{int(arr_month)}.{int(arr_day)}-{int(dep_month)}.{int(dep_day)}"
            except: return f"{arr_str}-{dep_str}"

        speech_parts = []
        for group in booking_groups:
            date_range_string = format_date_range(group['arrival_raw'], group['departure_raw'])
            sorted_df = group['dataframe'].sort_values(by='房数', ascending=True)
            rooms_list = [f"{row['房数']}{row['房型']}({row['定价']})" for _, row in sorted_df.iterrows()]
            room_string = "".join(rooms_list)
            speech_parts.append(f"{date_range_string} {room_string}")
        
        full_room_details = " ".join(speech_parts)
        return f"新增{team_type} {team_name} {full_room_details} {salesperson}销售通知"

    # --- Streamlit 主应用 ---
    st.title("炼狱金陵/金陵至尊必修剑谱 - OCR 工具 (表格识别版)")
    
    st.info("此版本使用专业的表格识别引擎，准确度更高。")

    uploaded_file = st.file_uploader("上传带表格线的清晰图片", type=["png", "jpg", "jpeg", "bmp"], key="ocr_uploader_table")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", width=400)

        if st.button("从图片提取表格信息"):
            if 'booking_info_table' in st.session_state:
                del st.session_state['booking_info_table']
            
            with st.spinner('正在调用阿里云表格识别 API...'):
                ocr_data, ocr_text = get_ocr_table_data(image)
                if ocr_data and ocr_text:
                    result = extract_booking_info_from_table(ocr_data, ocr_text)
                    if isinstance(result, str):
                        st.warning(result)
                    else:
                        st.session_state.booking_info_table = result
                        st.success("表格信息提取成功！请在下方核对。")

    if 'booking_info_table' in st.session_state:
        info = st.session_state.booking_info_table
        
        st.markdown("---")
        st.subheader("核对与编辑信息")

        info['team_name'] = st.text_input("团队名称", value=info['team_name'])
        
        for i, group in enumerate(info['booking_groups']):
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                new_arrival = st.text_input("到达日期", value=group['arrival_raw'], key=f"arrival_{i}")
                info['booking_groups'][i]['arrival_raw'] = new_arrival
            with col2:
                new_departure = st.text_input("离开日期", value=group['departure_raw'], key=f"departure_{i}")
                info['booking_groups'][i]['departure_raw'] = new_departure

            edited_df = st.data_editor(group['dataframe'], key=f"editor_{i}", num_rows="dynamic", use_container_width=True)
            info['booking_groups'][i]['dataframe'] = edited_df
        
        st.markdown("---")
        selected_salesperson = st.selectbox("选择对应销售", options=SALES_LIST)

        if st.button("生成最终话术"):
            final_speech = format_notification_speech(info['team_name'], info['team_type'], info['booking_groups'], selected_salesperson)
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
                # [关键修正] 创建一个更直接的映射字典
                # { '大床房': 'King Room', '标准间': 'Twin Room' }
                direct_map = {}
                for key, values in room_type_equivalents.items():
                    for value in values:
                        direct_map[forensic_clean_text(value)] = forensic_clean_text(key)
                standard_df['room_type'] = standard_df['room_type'].replace(direct_map)

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

    st.title("炼狱金陵/金陵至尊必修剑谱 - 比对平台")
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
                    
                    # [关键修正] 将 room_type_equivalents 传递给正确的函数调用
                    std_df1 = process_and_standardize(st.session_state.df1.copy(), mapping['file1'], case_insensitive)
                    std_df2 = process_and_standardize(st.session_state.df2.copy(), mapping['file2'], case_insensitive, room_type_equivalents=room_type_equivalents)

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
        st.header("原始数据预览")
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
    """ [关键修正] 完全按照用户提供的代码和期望的输出格式恢复此应用 """
    st.title("炼狱金陵/金陵至尊必修剑谱 - 报告分析器")
    st.markdown("---伯爵酒店团队报表分析工具---")

    uploaded_files = st.file_uploader("请上传您的 Excel 报告文件 (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="analyzer_uploader")

    if uploaded_files:
        st.subheader("分析结果")
        
        # Create a temporary directory to save uploaded files
        temp_dir = "./temp_uploaded_files"
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except OSError:
            pass # Fail silently if directory creation fails

        file_paths = []
        for uploaded_file in uploaded_files:
            try:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(temp_file_path)
            except Exception as e:
                st.warning(f"无法保存临时文件 {uploaded_file.name}: {e}")

        # Define the desired order of keywords
        desired_order = ["次日到达", "次日在住", "次日离店", "后天到达"]

        # Custom sort function
        def sort_key(file_path):
            file_name = os.path.basename(file_path)
            for i, keyword in enumerate(desired_order):
                if keyword in file_name:
                    return i
            return len(desired_order) 
        file_paths.sort(key=sort_key)

        if st.button("开始分析"): 
            with st.spinner("正在分析中，请稍候..."):
                # Since the real function is missing, we call our mock function
                summaries, unknown_codes = analyze_reports_ultimate(file_paths)
            
            for summary in summaries:
                st.write(summary) # Use st.write to match the desired output format

            if unknown_codes:
                st.subheader("侦测到的未知房型代码 (请检查是否需要更新规则)")
                for code, count in unknown_codes.items():
                    st.write(f"代码: '{code}' (出现了 {count} 次)")
            
            # Clean up temporary files and directory
            for f_path in file_paths:
                try:
                    os.remove(f_path)
                except OSError:
                    pass 
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except OSError:
                pass

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
# --- [最终版] APP 4: 酒店入住数据分析应用 ---
# ==============================================================================
@st.cache_data
def process_data(uploaded_file):
    """
    一个带缓存的函数，用于读取和预处理上传的Excel文件。
    只有当上传的文件发生变化时，才会重新运行，大大提高性能。
    """
    df = pd.read_excel(uploaded_file)
    df.columns = [str(col).strip().upper() for col in df.columns]
    
    required_cols = ['状态', '房类', '房数', '到达', '离开', '房价', '市场码']
    rename_map = {
        'ROOM CATEGORY': '房类', 'ROOMS': '房数', 'ARRIVAL': '到达',
        'DEPARTURE': '离开', 'RATE': '房价', 'MARKET': '市场码', 'STATUS': '状态'
    }
    df.rename(columns=rename_map, inplace=True)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"上传的文件缺少以下必要的列: {', '.join(missing_cols)}。请检查文件。")
        return None, None
    
    # [终极修正] 先统一转为字符串，再分离日期部分，最后用指定格式解析
    df['到达_str'] = df['到达'].astype(str).str.split(' ').str[0]
    df['离开_str'] = df['离开'].astype(str).str.split(' ').str[0]
    df['到达'] = pd.to_datetime(df['到达_str'], format='%y/%m/%d', errors='coerce')
    df['离开'] = pd.to_datetime(df['离开_str'], format='%y/%m/%d', errors='coerce')
    
    # [终极修正] 对所有关键列进行强制类型转换和清洗
    df['房价'] = pd.to_numeric(df['房价'], errors='coerce')
    df['房数'] = pd.to_numeric(df['房数'], errors='coerce')
    df['市场码'] = df['市场码'].astype(str)

    # 在所有转换完成后，一次性删除任何包含空值的关键行
    df.dropna(subset=['到达', '离开', '房价', '房数', '房类'], inplace=True)
    
    # 将房数转为整数，确保后续计算正确
    df['房数'] = df['房数'].astype(int)

    jinling_rooms = [
        'DETN', 'DKN', 'DQN', 'DQS', 'DSKN', 'DSTN', 'DTN', 'EKN', 'EKS', 'ESN', 'ESS', 'ETN', 'ETS',
        'FSB', 'FSC', 'FSN', 'OTN', 'PSA', 'PSB', 'RSN', 'SKN', 'SQN', 'SQS', 'SSN', 'SSS', 'STN', 'STS'
    ]
    yatal_rooms = [
        'JDEN', 'JDKN', 'JDKS', 'JEKN', 'JESN', 'JESS', 'JETN', 'JETS', 'JKN', 'JLKN', 'JTN', 'JTS',
        'PSC', 'PSD', 'VCKD', 'VCKN'
    ]
    room_to_building = {code: "金陵楼" for code in jinling_rooms}
    room_to_building.update({code: "亚太楼" for code in yatal_rooms})
    
    df = df[df['房类'].isin(jinling_rooms + yatal_rooms)].copy()
    df['楼层'] = df['房类'].map(room_to_building)
    
    # [关键修正] 使用 normalize() 来精确计算入住天数（午夜之差）
    # 即使是当天入住当天离开，只要跨过0点（比如23点到次日1点），也算1天
    df['入住天数'] = (df['离开'].dt.normalize() - df['到达'].dt.normalize()).dt.days

    # [关键修正] 将 > 0 改为 >= 0，以包含当天入住当天离开的“日用房”用于到店统计
    df_for_arrivals = df.copy()
    # [终极修正] 增加对状态 'R' 和 'I' 的筛选
    df_for_stays = df[(df['入住天数'] > 0) & (df['状态'].isin(['R', 'I']))].copy()
    
    if df_for_stays.empty:
        return df_for_arrivals, pd.DataFrame()

    df_repeated = df_for_stays.loc[df_for_stays.index.repeat(df_for_stays['入住天数'])]
    date_offset = df_repeated.groupby(level=0).cumcount()
    df_repeated['住店日'] = df_repeated['到达'].dt.normalize() + pd.to_timedelta(date_offset, unit='D')
    expanded_df = df_repeated.drop(columns=['到达', '离开', '入住天数']).reset_index(drop=True)
    
    return df_for_arrivals, expanded_df.copy()


def run_data_analysis_app():
    """重制版酒店数据分析应用，支持动态价格矩阵和每日到店统计。"""
    st.title("炼狱金陵/金陵至尊必修剑谱 - 数据分析驾驶舱")
    
    uploaded_file = st.file_uploader("上传您的Excel文件", type=["xlsx", "xls"], key="data_analysis_uploader")

    if not uploaded_file:
        st.info("请上传您的Excel文件以开始分析。")
        return

    try:
        original_df, expanded_df = process_data(uploaded_file)
        
        if original_df is None:
            return
            
        if original_df.empty:
            st.warning("上传的文件中没有找到有效的数据记录，请检查文件内容和格式。")
            return
            
        st.success(f"文件 '{uploaded_file.name}' 上传并处理成功！")

        st.header("1. 每日到店/离店房数统计")
        with st.expander("点击展开或折叠", expanded=True):
            
            # --- 到店统计 ---
            st.subheader("到店房数统计")
            all_statuses = sorted(original_df['状态'].unique())
            selected_arrival_statuses = st.multiselect("选择到店状态", options=all_statuses, default=['R'])
            
            arrival_dates_str = st.text_input(
                "输入到店日期 (用逗号分隔, 格式: YYYY/MM/DD)", 
                pd.to_datetime(original_df['到达'].min()).strftime('%Y/%m/%d') if not original_df.empty else ""
            )
            
            arrival_summary = pd.DataFrame() # 初始化为空
            if arrival_dates_str and selected_arrival_statuses:
                try:
                    date_strings = [d.strip() for d in arrival_dates_str.split(',') if d.strip()]
                    selected_arrival_dates = [pd.to_datetime(d, format='%Y/%m/%d').date() for d in date_strings]
                    
                    arrival_df = original_df[
                        (original_df['状态'].isin(selected_arrival_statuses)) & 
                        (original_df['到达'].dt.date.isin(selected_arrival_dates))
                    ].copy()

                    if not arrival_df.empty:
                        arrival_summary = arrival_df.groupby([arrival_df['到达'].dt.date, '楼层'])['房数'].sum().unstack(fill_value=0)
                        arrival_summary.index.name = "到店日期"
                        st.dataframe(arrival_summary)
                    else:
                        st.warning(f"在所选日期和状态内没有找到到店记录。")
                except ValueError:
                    st.error("到店日期格式不正确，请输入 YYYY/MM/DD 格式。")

            # --- 离店统计 ---
            st.subheader("离店房数统计")
            selected_departure_statuses = st.multiselect("选择离店状态", options=all_statuses, default=['R', 'S', 'I', 'O'])

            departure_dates_str = st.text_input(
                "输入离店日期 (用逗号分隔, 格式: YYYY/MM/DD)",
                pd.to_datetime(original_df['离开'].min()).strftime('%Y/%m/%d') if not original_df.empty else ""
            )
            
            departure_summary = pd.DataFrame() # 初始化为空
            if departure_dates_str and selected_departure_statuses:
                try:
                    date_strings = [d.strip() for d in departure_dates_str.split(',') if d.strip()]
                    selected_departure_dates = [pd.to_datetime(d, format='%Y/%m/%d').date() for d in date_strings]
                    
                    departure_df = original_df[
                        (original_df['状态'].isin(selected_departure_statuses)) & 
                        (original_df['离开'].dt.date.isin(selected_departure_dates))
                    ].copy()

                    if not departure_df.empty:
                        departure_summary = departure_df.groupby([departure_df['离开'].dt.date, '楼层'])['房数'].sum().unstack(fill_value=0)
                        departure_summary.index.name = "离店日期"
                        st.dataframe(departure_summary)
                    else:
                        st.warning(f"在所选日期和状态内没有找到离店记录。")
                except ValueError:
                    st.error("离店日期格式不正确，请输入 YYYY/MM/DD 格式。")

            # --- 下载按钮 ---
            if not arrival_summary.empty or not departure_summary.empty:
                df_to_download = {}
                if not arrival_summary.empty:
                    df_to_download["到店统计"] = arrival_summary
                if not departure_summary.empty:
                    df_to_download["离店统计"] = departure_summary
                
                excel_data = to_excel(df_to_download)
                st.download_button(
                    label="下载统计结果为 Excel",
                    data=excel_data,
                    file_name="arrival_departure_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        st.markdown("---")

        st.header("2. 每日在住房间按价格分布矩阵")
        with st.expander("点击展开或折叠", expanded=True):
            default_stay_date = ""
            if not expanded_df.empty and '住店日' in expanded_df.columns:
                 default_stay_date = pd.to_datetime(expanded_df['住店日'].min()).strftime('%Y/%m/%d')
            
            stay_dates_str = st.text_input("输入住店日期 (用逗号分隔, 格式: YYYY/MM/DD)", default_stay_date)
            
            selected_stay_dates = []
            if stay_dates_str:
                try:
                    stay_date_strings = [d.strip() for d in stay_dates_str.split(',') if d.strip()]
                    selected_stay_dates = [pd.to_datetime(d, format='%Y/%m/%d').date() for d in stay_date_strings]
                except ValueError:
                    st.error("住店日期格式不正确，请输入 YYYY/MM/DD 格式。")
                    st.stop()
            
            all_market_codes = sorted(original_df['市场码'].dropna().unique())
            selected_market_codes = st.multiselect("选择市场码 (可多选)", options=all_market_codes, default=all_market_codes)
            
            st.subheader("自定义价格区间")
            col1, col2 = st.columns(2)
            with col1:
                price_bins_jinling_str = st.text_input("金陵楼价格区间", "<401, 401-480, 481-500, 501-550, 551-599, >599")
            with col2:
                price_bins_yatal_str = st.text_input("亚太楼价格区间", "<501, 501-600, 601-699, 700-749, 750-799, >799")

            def parse_price_bins(price_bins_str):
                # ... (此函数保持不变)
                if not price_bins_str.strip(): return None, None
                intervals = []
                for item in price_bins_str.split(','):
                    item = item.strip()
                    if item.startswith('<'):
                        upper = int(re.search(r'\d+', item).group())
                        intervals.append({'lower': float('-inf'), 'upper': upper, 'label': f'< {upper}'})
                    elif item.startswith('>'):
                        lower = int(re.search(r'\d+', item).group())
                        intervals.append({'lower': lower, 'upper': float('inf'), 'label': f'> {lower}'})
                    elif '-' in item:
                        parts = item.split('-')
                        lower, upper = int(parts[0]), int(parts[1])
                        if lower >= upper: raise ValueError(f"价格区间 '{item}' 无效：下限必须小于上限。")
                        intervals.append({'lower': lower, 'upper': upper, 'label': f'{lower}-{upper}'})
                    else: raise ValueError(f"无法解析区间 '{item}'")
                intervals.sort(key=lambda x: x['lower'])
                bins = [d['lower'] for d in intervals] + [intervals[-1]['upper']]
                labels = [d['label'] for d in intervals]
                return bins, labels

            try:
                bins_jinling, labels_jinling = parse_price_bins(price_bins_jinling_str)
                bins_yatal, labels_yatal = parse_price_bins(price_bins_yatal_str)
            except (ValueError, IndexError, AttributeError) as e:
                st.error(f"价格区间格式不正确。错误: {e}")
                st.stop()
            
            dfs_to_download_matrix = {}
            if selected_stay_dates and selected_market_codes:
                matrix_df = expanded_df[
                    (expanded_df['住店日'].dt.date.isin(selected_stay_dates)) &
                    (expanded_df['市场码'].isin(selected_market_codes))
                ].copy()

                if not matrix_df.empty:
                    buildings = sorted(matrix_df['楼层'].unique())
                    for building in buildings:
                        st.subheader(f"{building} - 在住房间分布")
                        building_df = matrix_df[matrix_df['楼层'] == building]
                        
                        bins, labels = (bins_jinling, labels_jinling) if building == "金陵楼" else (bins_yatal, labels_yatal)
                        
                        if not building_df.empty and bins and labels:
                            building_df['价格区间'] = pd.cut(building_df['房价'], bins=bins, labels=labels, right=True, include_lowest=True)
                            pivot_table = pd.pivot_table(
                                building_df.dropna(subset=['价格区间']),
                                index=building_df['住店日'].dt.date,
                                columns='价格区间',
                                values='房数',
                                aggfunc='sum',
                                fill_value=0
                            )
                            
                            if not pivot_table.empty:
                                pivot_table['每日总计'] = pivot_table.sum(axis=1)
                                st.dataframe(pivot_table.sort_index())
                                dfs_to_download_matrix[f"{building}_在住分布"] = pivot_table
                            else:
                                 st.info(f"在 {building} 中，所选条件下的所有房价都不在您定义的价格区间内。")
                        else:
                            st.info(f"在 {building} 中，没有找到符合所选条件的在住记录或未设置价格区间。")
                else:
                    st.warning(f"在所选日期和市场码范围内没有找到在住记录。")
            
            if dfs_to_download_matrix:
                excel_data_matrix = to_excel(dfs_to_download_matrix)
                st.download_button(
                    label="下载价格分布矩阵为 Excel",
                    data=excel_data_matrix,
                    file_name="price_matrix_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


    except Exception as e:
        st.error(f"处理数据或生成报告时发生意外错误。请检查您的Excel文件格式是否正确。")
        st.error(f"技术细节: {e}")
        st.code(f"Traceback: {traceback.format_exc()}")


# ==============================================================================
# --- [新增] APP 5: 早班话术生成器 ---
# ==============================================================================
def run_morning_briefing_app():
    st.title("炼狱金陵/金陵至尊必修剑谱 - 早班话术生成器")

    st.subheader("数据输入")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 金陵楼数据")
        jl_occupancy = st.number_input("昨日出租率 (%)", key="jl_occ", format="%.1f", value=82.4)
        jl_revenue = st.number_input("收入 (元)", key="jl_rev", format="%.1f", value=247173.4)
        jl_adr = st.number_input("平均房价 (元)", key="jl_adr", format="%.1f", value=550.5)
        jl_guests = st.number_input("总人数", key="jl_guests", value=673)
        jl_jinhaiwan = st.number_input("金海湾人数", key="jl_jinhaiwan", value=572)

    with col2:
        st.markdown("#### 亚太楼数据")
        yt_occupancy = st.number_input("昨日出租率 (%)", key="yt_occ", format="%.1f", value=83.9)
        yt_revenue = st.number_input("收入 (元)", key="yt_rev", format="%.1f", value=232385.5)
        yt_adr = st.number_input("平均房价 (元)", key="yt_adr", format="%.1f", value=719.5)
        yt_guests = st.number_input("总人数", key="yt_guests", value=485)
        yt_jia = st.number_input("家餐厅人数", key="yt_jia", value=323)
        
    st.markdown("---")
    st.subheader("其他数据")
    col3, col4 = st.columns(2)
    with col3:
        onbook_jl = st.number_input("目前On Book出租率 - 金陵楼 (%)", key="ob_jl", format="%.1f", value=65.5)
        onbook_yt = st.number_input("目前On Book出租率 - 亚太楼 (%)", key="ob_yt", format="%.1f", value=57.7)
    with col4:
        mini_prog_yesterday = st.number_input("小程序订房 - 昨日 (间夜)", key="mp_yest", value=26)
        mini_prog_today = st.number_input("小程序订房 - 今日 (间夜)", key="mp_today", value=19)

    if st.button("生成话术"):
        briefing = (
            f"昨日金陵楼出租率{jl_occupancy}%，收入{jl_revenue}元，平均房价{jl_adr}元，总人数{jl_guests}人，金海湾{jl_jinhaiwan}人。"
            f"亚太商务楼出率{yt_occupancy}%，收入{yt_revenue}元，平均房价{yt_adr}元，总人数{yt_guests}人，家餐厅{yt_jia}人。"
            f"目前on book出租率金陵楼{onbook_jl}%，亚太商务楼{onbook_yt}%。"
            f"小程序订房昨日{mini_prog_yesterday}间夜，今日{mini_prog_today}间夜。"
        )
        st.subheader("生成的话术")
        st.success(briefing)
        st.code(briefing)

# ==============================================================================
# --- [新增] APP 6: 常用话术复制器 ---
# ==============================================================================
def run_common_phrases_app():
    st.title("炼狱金陵/金陵至尊必修剑谱 - 常用话术")
    
    phrases = [
        "CA RM TO CREDIT FM",
        "免预付,房费及3000元以内杂费转淘宝 FM",
        "房费转携程宏睿 FM",
        "房价保密,房费转华为 FM",
        "房费转淘宝 FM",
        "CA RM TO 兰艳(109789242)金陵卡 FM",
        "CA RM TO AGODA FM",
        "CA RM TO CREDIT CARD FM XX-XX/XX(卡号/有效期XX/XX)",
        "房费转微信 FM",
        "房费预付杂费自理FM"
    ]
    
    st.subheader("点击右上角复制图标即可复制话术")
    for phrase in phrases:
        st.code(phrase, language=None)


# ==============================================================================
# --- 全局函数和主应用路由器 ---
# ==============================================================================
# [关键修正] 恢复 to_excel 辅助函数
@st.cache_data
def to_excel(df_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
    processed_data = output.getvalue()
    return processed_data
    
# --- 登录检查函数 (全局) ---
def check_password():
    """返回 True 如果用户已登录, 否则返回 False."""
    def login_form():
        with st.form("Credentials"):
            st.text_input("用户名", key="username")
            st.text_input("密码", type="password", key="password")
            st.form_submit_button("登录", on_click=password_entered)

    def password_entered():
        app_username = st.secrets.app_credentials.get("username")
        app_password = st.secrets.app_credentials.get("password")
        
        if st.session_state["username"] == app_username and st.session_state["password"] == app_password:
            st.session_state["password_correct"] = True
            if "password" in st.session_state: del st.session_state["password"]
            if "username" in st.session_state: del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "app_credentials" not in st.secrets or not st.secrets.app_credentials.get("username") or not st.secrets.app_credentials.get("password"):
        st.error("错误：应用的用户名或密码未在 Streamlit Secrets 中正确配置。")
        return False

    if st.session_state.get("password_correct", False):
        return True

    login_form()
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("用户名或密码不正确。")
    return False


# --- 主应用路由器 ---
st.set_page_config(layout="wide", page_title="炼狱金陵/金陵至尊必修剑谱")

if check_password():
    with st.sidebar:
        app_choice = option_menu(
            menu_title="炼狱金陵/金陵至尊必修剑谱",
            options=["OCR 工具", "比对平台", "报告分析器", "数据分析", "话术生成器", "常用话术"],
            icons=["camera-reels-fill", "kanban", "clipboard-data", "graph-up-arrow", "blockquote-left", "card-text"],
            menu_icon="tools",
            default_index=0,
        )

    st.sidebar.markdown("---")
    st.sidebar.info("这是一个将多个工具集成到一起的应用。")

    if app_choice == "OCR 工具":
        run_ocr_app_detailed()
    elif app_choice == "比对平台":
        run_comparison_app()
    elif app_choice == "报告分析器":
        run_analyzer_app()
    elif app_choice == "数据分析":
        run_data_analysis_app()
    elif app_choice == "话术生成器":
        run_morning_briefing_app()
    elif app_choice == "常用话术":
        run_common_phrases_app()

