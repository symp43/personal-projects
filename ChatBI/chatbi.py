from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from io import BytesIO
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain.agents import create_react_agent, AgentExecutor
import pymysql
from decimal import Decimal
from langchain.agents import Tool
import json
from langchain.agents.output_parsers.react_json_single_input import ReActJsonSingleInputOutputParser
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.pydantic_v1 import BaseModel, Field
from pyecharts.charts import Bar, Line, Pie
from pyecharts import options as opts
from typing import List, Union, Tuple, Any
from streamlit_echarts import st_echarts
import time
import uuid
from pathlib import Path
import os
import re
import base64
from PIL import Image as PILImage
from dotenv import load_dotenv
import datetime
import chromadb
import warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


# 加载环境变量
load_dotenv()

# 获取base64图标函数
def get_base64_icon(icon_path):
    with open(icon_path, "rb") as icon_file:
        return base64.b64encode(icon_file.read()).decode()

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        # print(f"文件夹 '{path}' 已创建")
    else:
        pass
        # print(f"文件夹 '{path}' 已存在")

# 从环境变量获取用户名和密码
VALID_USERNAME = os.getenv("BI_USERNAME")
VALID_PASSWORD = os.getenv("BI_PASSWORD")

BASE_DIR = Path(__file__).resolve().parent
BASE_URL_IMAGES = os.getenv("BASE_URL_IMAGES")

BASE_PATH_IMAGES = os.path.join(BASE_DIR, "images")
# 判断图片存储文件夹是否存在，不存在则创建
make_dir(BASE_PATH_IMAGES)

TMP_DIR = os.path.join(BASE_DIR, "tmp")
# 判断临时文件夹是否存在，不存在则创建
make_dir(TMP_DIR)


# 测试数据库链接
def test_database_connection(db_config):

    try:
        # 尝试建立数据库连接
        connection = pymysql.connect(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            port=int(db_config["port"]),
            charset=db_config["charset"]
        )
        #如果连接成功，关闭连接并返回True
        connection.close()
        return True, "✅ 数据库连接成功！"
    except pymysql.MySQLError as e:
        # 连接失败，返回错误信息
        return False, f"❌ 数据库连接失败: {str(e)}"
    except Exception as e:
        # 其他异常
        return False, f"❌ 发生未知错误: {str(e)}"

# 读取默认文档
def read_default_file(file_path: str):
    with open(file_path, "rb") as f:
        return f.read()

def generate_img_filename(type: str):
    # 生成随机的UUID（去掉连字符）
    random_uuid = str(uuid.uuid4()).replace('-', '')

    # 获取当前时间戳（精确到毫秒）
    timestamp = int(time.time() * 1000)

    # 组合成文件名
    html_filename = f"{type}_{random_uuid}_{timestamp}.html"
    png_filename = f"{type}_{random_uuid}_{timestamp}.png"
    html_path = os.path.join(BASE_PATH_IMAGES, html_filename)
    png_path = os.path.join(BASE_PATH_IMAGES, png_filename)
    png_url = f"{BASE_URL_IMAGES}/{png_filename}"

    return html_path, png_path, png_url

def delete_file_if_is_file(filename):

    if os.path.exists(filename) and os.path.isfile(filename):
        os.remove(filename)
        print(f"文件 {filename} 已删除")
    else:
        print(f"{filename} 不是文件或不存在")


# 工具一：sql查询
def get_sql_result(sql_query: str):

    db_config = st.session_state.db_config
    connection = pymysql.connect(
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"],
        port=int(db_config["port"]),
        charset=db_config["charset"]
    )
    print("sql_query:", sql_query)
    try:
        with connection.cursor() as cursor:
            sql = sql_query
            cursor.execute(sql)
            results = cursor.fetchall()

            # 自定义序列化函数
            def default_serializer(obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                elif isinstance(obj, (datetime.date, datetime.datetime)):
                    return obj.isoformat()
                elif isinstance(obj, bytes):
                    return obj.decode('utf-8')
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            # 处理结果
            processed_results = []
            for row in results:
                processed_row = []
                for item in row:
                    try:
                        # 尝试直接序列化
                        json.dumps(item, default=default_serializer)
                        processed_row.append(item)
                    except (TypeError, ValueError):
                        # 无法序列化的值使用自定义函数处理
                        processed_row.append(default_serializer(item))
                processed_results.append(tuple(processed_row))

            # 使用自定义序列化器
            return json.dumps(processed_results, default=default_serializer)
    except Exception as e:
        # 捕获其他异常
        error_info = {
            "status": "error",
            "error_message": str(e)
        }
        return json.dumps(error_info)
    finally:
        connection.close()

sqlTool = Tool(
    name="查询数据库",
    description="数据库sql查询函数",
    func=get_sql_result,
)

# 定义图表参数的通用 Schema
class ChartInputSchema(BaseModel):
    title: str = Field(description="图表标题")
    xaxis_name: str = Field(description="X轴名称")
    xaxis_data: List[str] = Field(description="X轴数据列表")

    series_names: Union[str, List[str]] = Field(
        description="数据系列名称。如果是单系列，可以是字符串 '销售额'；如果是多系列，必须是列表 ['销售额', '利润']"
    )

    series_data: Union[List[float], List[List[float]]] = Field(
        description="数据内容。单系列传 [10, 20]；多系列传 [[10, 20], [15, 30]]"
    )

# 辅助函数：标准化数据格式
def normalize_chart_inputs(names: Union[str, List[str]], data: Union[List[float], List[List[float]]]):

    # 处理 names：如果是字符串，转成列表
    if isinstance(names, str):
        names = [names]
    # 处理 data：
    if data and isinstance(data[0], (int, float)):
        data = [data]
    return names, data

# 工具二：柱状图工具
def generate_bar_chart(title: str, xaxis_name: str, xaxis_data: List[str], series_names: Union[str, List[str]],
                       series_data: Union[List[float], List[List[float]]]):
    # 调用辅助函数
    norm_names, norm_data = normalize_chart_inputs(series_names, series_data)

    # 创建柱状图
    bar = (
        Bar()
        .add_xaxis(xaxis_data)
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            toolbox_opts=opts.ToolboxOpts(),
            legend_opts=opts.LegendOpts(pos_left="right"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),  # 开启提示框
            xaxis_opts=opts.AxisOpts(name=xaxis_name),
            yaxis_opts=opts.AxisOpts(name="")
        )
    )

    # 循环添加多个数据系列
    for name, data in zip(norm_names, norm_data):
        bar.add_yaxis(name, data)

    chart_json = bar.dump_options_with_quotes()
    # 生成一个唯一ID，存入 session_state
    chart_id = f"chart_{uuid.uuid4()}"
    if "chart_data_buffer" not in st.session_state:
        st.session_state.chart_data_buffer = {}
    st.session_state.chart_data_buffer[chart_id] = chart_json
    return f"CHART_RENDER_TRIGGER:{chart_id}"

# 多参数函数，需要用 StructuredTool
barChartTool = StructuredTool.from_function(
    func=generate_bar_chart,
    name="生成柱状图",
    description="生成可视化柱状图表，支持单或多系列对比",
    args_schema=ChartInputSchema
)

# 工具三：折线图工具
def generate_line_chart(title: str, xaxis_name: str, xaxis_data: List[str], series_names: Union[str, List[str]],
                        series_data: Union[List[float], List[List[float]]]):
    # 标准化数据
    norm_names, norm_data = normalize_chart_inputs(series_names, series_data)

    # 创建折线图
    line = (
        Line()
        .add_xaxis(xaxis_data)
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            toolbox_opts=opts.ToolboxOpts(),
            legend_opts=opts.LegendOpts(pos_left="right"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),  # 开启提示框
            xaxis_opts=opts.AxisOpts(name=xaxis_name),
            yaxis_opts=opts.AxisOpts(name="")
        )
    )

    for name, data in zip(norm_names, norm_data):
        line.add_yaxis(name, data, is_smooth=True, symbol_size=8)

    chart_json = line.dump_options_with_quotes()
    # 生成一个唯一ID，存入 session_state
    chart_id = f"chart_{uuid.uuid4()}"
    if "chart_data_buffer" not in st.session_state:
        st.session_state.chart_data_buffer = {}
    st.session_state.chart_data_buffer[chart_id] = chart_json
    return f"CHART_RENDER_TRIGGER:{chart_id}"

# 多参数函数，需要用 StructuredTool
lineChartTool = StructuredTool.from_function(
    func=generate_line_chart,
    name="生成折线图",
    description="生成可视化折线图表，支持单或多条折线对比",
    args_schema=ChartInputSchema
)

# 工具四：饼图
def generate_pie_chart(title: str, data_pair: List[List[Union[str, float]]]):
    pie = (
        Pie()
        .add(
            series_name="",
            data_pair=data_pair,
            radius=["40%", "70%"],  # 环形图更好看
            center=["50%", "50%"],
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(pos_left="right", orient="vertical"),
            tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c} ({d}%)")
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(formatter="{b}: {d}%")
        )
    )

    chart_json = pie.dump_options_with_quotes()
    # 生成一个唯一ID，存入 session_state
    chart_id = f"chart_{uuid.uuid4()}"
    if "chart_data_buffer" not in st.session_state:
        st.session_state.chart_data_buffer = {}
    st.session_state.chart_data_buffer[chart_id] = chart_json
    return f"CHART_RENDER_TRIGGER:{chart_id}"

pieChartTool = StructuredTool.from_function(
    func=generate_pie_chart,
    name="生成饼图",
    description="生成可视化饼图表",
)

# 设置页面配置
icon_base64 = os.path.join(BASE_DIR, "docs_imgs", "favicon.ico")
st.set_page_config(
    page_title="XXXX智能问答BI",
    page_icon=f"data:image/x-icon;base64,{icon_base64}",
    layout="wide"
)

# 自定义CSS样式
login_css = """
<style>
    /* 使用更直接的选择器来设置登录页面背景 */
    .stApp {
        background: linear-gradient(120deg, #43e97b 0%, #38f9d7 100%);
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* 登录页面标题样式 */
    .center-title {
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* 登录框样式 */
    .login-container {
        background: transparent;
        padding: 0;
        border-radius: 0;
        box-shadow: none;
        backdrop-filter: none;
        border: none;
        max-width: 200px;
        margin: 0 auto;
    }

    /* 登录页面输入框样式 */
    .stTextInput>div>div>input, 
    .stTextInput>div>div>input:focus {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px;
    }

    /* 登录页面按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(120deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px;
        margin-top: 1rem;
    }

    /* 错误信息样式 */
    .error-message {
        color: #ff4b4b;
        text-align: center;
        margin-top: 1rem;
        font-weight: bold;
    }
</style>
"""
reset_css = """
<style>
    .stApp {
        background: none;
        background-color: white; /* 或者你想要的默认背景色 */
        background-image: none;
    }
</style>
"""

# 初始化session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'login_error' not in st.session_state:
    st.session_state.login_error = False


# 登录函数
def login(username, password):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        st.session_state.logged_in = True
        st.session_state.login_error = False
        st.rerun()
    else:
        st.session_state.login_error = True


# 将登录页面的渲染逻辑封装成函数（推荐，代码更整洁）
def show_login_page(container):
    # 应用登录页面CSS
    st.markdown(login_css, unsafe_allow_html=True)

    with container.container():
        # 显示登录页面标题
        st.markdown('<h1 class="center-title">XXXX智能问答BI<br>登录页面</h1>', unsafe_allow_html=True)

        # 创建一个居中的列
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.subheader("🔐 用户登录")

            with st.form("login_form"):
                username = st.text_input("👤 用户名", placeholder="请输入用户名")
                password = st.text_input("🔒 密码", type="password", placeholder="请输入密码")
                login_button = st.form_submit_button("登录", use_container_width=True)

                if login_button:
                    if username and password:
                        login(username, password)
                    else:
                        st.session_state.login_error = True
                        st.error("请输入用户名和密码")

            if st.session_state.login_error:
                st.markdown('<div class="error-message">❌ 用户名或密码错误，请重试</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # 页脚信息
        st.markdown("""
                <div style="text-align: center; color: black; margin-top: 15rem;">
                    <p>XXXX - 智能问答BI系统</p>
                </div>
                """, unsafe_allow_html=True)

# 主应用逻辑
def main():
    # 创建一个主占位符，所有的页面内容都放在这里
    main_placeholder = st.empty()

    if not st.session_state.logged_in:
        # 在占位符中渲染登录页
        show_login_page(main_placeholder)
    else:
        # 1. 首先清空占位符，确保登录页元素被移除
        main_placeholder.empty()
        # 2. 立即注入重置CSS，覆盖掉登录页的背景
        st.markdown(reset_css, unsafe_allow_html=True)
        # 登录成功后显示主应用
        st.title("XXXX智能问答系统")

        # 数据库配置默认值
        DEFAULT_DB_CONFIG = {
            "host": os.getenv("MYSQL_HOST"),
            "port": os.getenv("MYSQL_PORT"),
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE"),
            "charset": os.getenv("MYSQL_CHARSET")
        }

        # 初始化数据库配置
        if "db_config" not in st.session_state:
            st.session_state.db_config = DEFAULT_DB_CONFIG.copy()

        # 默认文件路径
        DEFAULT_RAG_FILE_PATH = os.path.join(BASE_DIR, "参考知识库.txt")
        # 默认数据字典文件路径
        DEFAULT_DATA_DIC_FILE_PATH = os.path.join(BASE_DIR, "参考数据字典.txt")


        # 侧边栏分成3个区块
        with st.sidebar:
            # 区块1: 文档上传区
            with st.expander("📁 知识库文档上传", expanded=False):
                # 检索文档部分
                uploaded_files = st.file_uploader(
                    label="上传知识库文档", type=["txt"], accept_multiple_files=True
                )

                # 如果没有上传文件，则使用默认文件
                if not uploaded_files:
                    default_file_content = read_default_file(DEFAULT_RAG_FILE_PATH)
                    default_file = BytesIO(default_file_content)
                    default_file.name = "参考知识库.txt"
                    uploaded_files = [default_file]
                    # print("未上传文件：")
                    # print(default_file)
                    # 如果没有上传文件，需要强制清除一下缓存
                    # 因为当用户先上传文件，再取消之后，streamlit不会自动清除缓存，会出bug
                    # 用户初次访问，会执行 configure_retriever，用户上传文件之后，会执行 configure_retriever
                    # 但是此时用户再删除文件，便不会重新执行 configure_retriever，按理来说应该要执行的，因为函数参数已经更新了，但实际上并没有执行
                    # 这就导致 configure_retriever用的还是之前缓存的那个，而不是变更之后的，就会报错
                    # 所以我们手动清除缓存
                    st.cache_resource.clear()  # 清除所有 @st.cache_resource 缓存
                    # st.rerun()  # 重新运行应用
                    st.info("当前未上传检索文档，使用默认文档：参考知识库.txt")
                    if st.button("🔍 预览默认检索文档"):
                        with st.expander("📄 默认检索文档内容", expanded=True):
                            st.code(default_file_content.decode('utf8'), language="text")
            # 区域2：数据字典文档上传
            with st.expander("📖 数据字典文档上传", expanded=False):
                # 数据字典部分
                data_uploaded_files = st.file_uploader(
                    label="上传数据字典文档", type=["txt"], accept_multiple_files=True
                )

                # 如果没有上传文件，则使用默认文件
                if not data_uploaded_files:
                    default_file_content = read_default_file(DEFAULT_DATA_DIC_FILE_PATH)
                    default_file = BytesIO(default_file_content)
                    default_file.name = "参考数据字典.txt"
                    data_uploaded_files = [default_file]
                    st.cache_resource.clear()  # 清除所有 @st.cache_resource 缓存
                    st.info("当前未上传数据字典文档，使用默认文档：参考数据字典.txt")
                    if st.button("📊 预览默认数据字典"):
                        with st.expander("📝 默认数据字典内容", expanded=True):
                            st.code(default_file_content.decode('utf8'), language="text")

            # 区块3: 数据库配置区
            with st.expander("🗄 数据库配置", expanded=False):
                if st.button("⚙️ 配置数据库连接"):
                    st.session_state.show_db_config = True

                if st.session_state.get("show_db_config", False):
                    st.write("请填写数据库连接信息:")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.session_state.db_config["host"] = st.text_input(
                            "主机地址",
                            value=st.session_state.db_config["host"],
                            help="数据库服务器地址"
                        )
                        st.session_state.db_config["port"] = st.text_input(
                            "端口",
                            value=st.session_state.db_config["port"],
                            help="数据库端口号"
                        )
                        st.session_state.db_config["user"] = st.text_input(
                            "用户名",
                            value=st.session_state.db_config["user"],
                            help="数据库用户名"
                        )

                    with col2:
                        st.session_state.db_config["password"] = st.text_input(
                            "密码",
                            value=st.session_state.db_config["password"],
                            type="password",
                            help="数据库密码"
                        )
                        st.session_state.db_config["database"] = st.text_input(
                            "数据库名",
                            value=st.session_state.db_config["database"],
                            help="要连接的数据库名称"
                        )
                        st.session_state.db_config["charset"] = st.text_input(
                            "字符集",
                            value=st.session_state.db_config["charset"],
                            help="数据库字符集"
                        )

                    test_col1, test_col2 = st.columns([1, 3])
                    with test_col1:
                        if st.button("🔍 测试连接"):
                            success, message = test_database_connection(st.session_state.db_config)
                            test_col2.write(message)
                            st.session_state.db_connection_success = success

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("✅ 保存配置", disabled=not st.session_state.get("db_connection_success", False)):
                            st.session_state.show_db_config = False
                            st.rerun()
                    with col2:
                        if st.button("🔄 重置默认"):
                            st.session_state.db_config = DEFAULT_DB_CONFIG.copy()
                            st.session_state.db_connection_success = False
                            st.rerun()
                    with col3:
                        if st.button("❌ 取消"):
                            st.session_state.show_db_config = False
                            st.rerun()

                    if not st.session_state.get("db_connection_success", False):
                        st.warning("请先测试数据库连接，成功后才能保存配置")

            # 区块4: 帮助说明区
            with st.expander("❓ 图表使用说明", expanded=True):
                st.markdown("""
                **支持自动生成的图表类型：**
                - 📊 柱状图 - 数据对比
                - 📈 折线图 - 趋势分析  
                - 🥧 饼图 - 比例分布

                **示例问题：**
                - 销量前几名的产品占比
                - 某产品的销售趋势分析
                """)


        # 百川向量模型Key
        BAICHUAN_EMBEDDINGS_KEY = os.getenv("BAICHUAN_EMBEDDINGS_KEY")

        # 这里是缓存retriever
        # Streamlit 会在每次用户交互时重新运行整个脚本，为了避免重复计算昂贵的资源（如数据库连接、模型加载等），可以使用缓存装饰器
        # 以下情况会重新执行该函数
        # 1、TTL过期：代码设置了ttl="1h"，即从上次执行开始1小时后，缓存会失效，函数会重新执行。
        # 2、输入参数变更：当函数参数 uploaded_files 发生变化时：用户上传了新文件等
        @st.cache_resource(ttl="1h")
        def configure_retriever(uploaded_files):
            # print("configure_retriever函数执行开始：")
            docs = []
            temp_dir = tempfile.TemporaryDirectory(dir=TMP_DIR)
            for file in uploaded_files:
                temp_filepath = os.path.join(temp_dir.name, file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(file.getvalue())
                loader = TextLoader(temp_filepath, encoding="utf-8")
                docs.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
            splits = text_splitter.split_documents(docs)

            key = BAICHUAN_EMBEDDINGS_KEY
            embeddings = BaichuanTextEmbeddings(api_key=key)

            # streamlit刷新一次，代码就会重新执行一次，就会把文件拿出来重新插入向量数据库，就会导致向量数据库的数据不断重复
            # 上面临时文件读取完毕之后，就会被删除，所有的数据都存在了这个向量数据库里
            # 而这个向量数据库如果不实体化，他是默认在内存中的
            # 我们多次创建文档，他默认是在内存里的同一个地方，所以我们如果多次执行，数据就会在同一个集合里不断叠加，造成重复
            # 这里没有指定 persist_directory
            # 但 Chroma 的默认行为是：
            # 在内存中创建一个临时集合（collection），但这个集合会被自动分配一个固定的默认名称（通常是 "langchain"）
            # 即使你重新执行 from_documents()，只要 Python 进程没有完全重启（比如在 Streamlit 的连续交互中）
            # Chroma 的客户端会继续连接到同一个内存数据库，并向同名集合追加数据
            # vectordb = Chroma.from_documents(splits, embeddings)

            # 每次执行先删除集合，否则多次执行会向同一个集合追加数据 造成重复
            client = chromadb.Client()
            try:
                # 这个是默认的集合名字
                client.delete_collection(Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME)  # 删除默认集合
            except Exception as e:
                print(f"删除集合出错（可能无害）: {e}")

            # 关键步骤2：创建新集合（无需指定名称，Chroma自动生成）
            vectordb = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                client=client
            )

            collection = vectordb._client.get_collection(vectordb._collection.name)
            records = collection.get()  # 获取所有数据
            print(f"向量数据库总存储数: {len(records['ids'])}")

            retriever = vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 2}
            )
            return retriever

        retriever = configure_retriever(uploaded_files)

        retriever_tool = create_retriever_tool(
            retriever=retriever,
            name="知识库检索",
            description="用于检索用户提出的问题，并基于检索到的文档内容进行回复.",
        )

        tools = [sqlTool, barChartTool, lineChartTool, pieChartTool, retriever_tool]

        # 记忆和提示词配置
        msgs = StreamlitChatMessageHistory()
        # chat_memory就是具体存数据的地方   不指定默认存在内存，即InMemoryChatMessageHistory里
        memory = ConversationBufferMemory(
            chat_memory=msgs, memory_key="chat_history", output_key="output"
        )

        @st.cache_resource(ttl="1h")
        def read_data_dictionary(data_uploaded_files):
            docs = []
            for file in data_uploaded_files:
                docs.append(file.getvalue().decode("utf-8"))
            return '\n'.join(docs)

        data_dictionary = "数据库表信息如下：\n" + read_data_dictionary(data_uploaded_files)

        instructions = """
        你是一个专业的问题解决助手，具备以下核心能力：
        1、数据库查询 - 直接访问结构化数据 
        2、RAG检索 - 从知识库获取最新信息
        3、可视化生成 - 创建数据图表(仅支持柱状图、饼图、折线图)

        同时你也是一个零售业数据分析专家，请基于提供的数据进行深度分析：
        1. 销售业绩分析
        2. 商品结构分析
        3. 顾客行为分析

        ！！！注意：1.如果工具返回了 'CHART_RENDER_TRIGGER:...' 格式的内容，请原样输出该字符串作为 Final Answer，不要做任何修改或包裹额外的 Markdown 标记。
        2.必须严格按照ReAct格式输出，每个步骤都要有Thought、Action、Observation
        3.在最后一步必须使用"Final Answer:"开头来给出最终答案，不要在任何中间步骤输出最终答案

        工作流程：
        明确用户意图，智能判断最适合的工具组合：
        需要背景知识/最新信息 → RAG检索
        需要精确数据 → 数据库查询
        需要数据呈现 → 图表生成
        必要时进行多工具协同（如：先用RAG确认概念，再查询具体数据）
        如果你从文档或者数据库查询结果中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
        """

        # 基础提示模板（更新以包含图表工具）
        base_prompt_template = """
        {instructions}

        {data_dictionary}

        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        The way you use the tools is by specifying a json blob.
        Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

        The only values that should be in the "action" field are: {tool_names}

        The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

        ```
        {{
          "action": $TOOL_NAME,
          "action_input": $INPUT
        }}
        ```

        请严格按照上述格式来生成，$JSON_BLOB前后都必须用 ``` 包裹，不要忘记! important

        The $JSON_BLOB should contain a SINGLE action with properly named parameters, even for single-parameter tools. Always include the parameter names explicitly. Here are examples:
        For multi-parameter tools

        你必须严格遵守JSON格式规范生成$JSON_BLOB。特别注意：
        - 必须包含完整的闭合括号

        ALWAYS use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action:
        ```
        $JSON_BLOB
        ```

        Observation: the result of the actionr
        ... (this Thought/Action/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin! Reminder to always use the exact characters `Final Answer` when responding.

        Previous conversation history:
        {chat_history}

        New input: {input}
        {agent_scratchpad}


        """

        base_prompt = PromptTemplate.from_template(base_prompt_template)
        prompt = base_prompt.partial(instructions=instructions, data_dictionary=data_dictionary)

        # 创建llm
        llm = ChatOpenAI(model=os.getenv("LLM_MODEL_NAME"), openai_api_key=os.getenv("LLM_API_KEY"),
                         openai_api_base=os.getenv("LLM_BASE_URL"))

        # 创建agent
        agent = create_react_agent(llm, tools, prompt, output_parser=ReActJsonSingleInputOutputParser())
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                                       handle_parsing_errors=True)

        # 初始化消息状态
        if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "您好，我是ChatBI智能助手，我可以查询文档，查询数据库，并为您生成图表"}]
            # 重置内存
            if 'memory' in globals():
                memory.clear()

        # 加载历史聊天记录
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # 用户输入处理
        user_query = st.chat_input(placeholder="请开始提问吧!")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                with st.status("🤖 正在思考与检索...", expanded=True) as status:
                    st_cb = StreamlitCallbackHandler(st.container())
                    config = {"callbacks": [st_cb]}
                    response = agent_executor.invoke({"input": user_query}, config=config)
                    status.update(label="✅ 回答完成", state="complete", expanded=False)

                output_text = response["output"]

                # 检查是否有图表触发标记
                chart_ids = re.findall(r'CHART_RENDER_TRIGGER:(chart_[\w-]+)', output_text)

                # 清理文本，把标记去掉，只显示 LLM 的分析话术
                clean_text = re.sub(r'CHART_RENDER_TRIGGER:chart_[\w-]+', '', output_text).strip()
                if clean_text:
                    st.write(clean_text)

                # 从缓存中取出数据进行渲染
                for c_id in chart_ids:
                    # 从 session_state 获取真实的 JSON 数据
                    chart_json = st.session_state.chart_data_buffer.get(c_id)
                    if chart_json:
                        options = json.loads(chart_json)
                        st_echarts(options=options, height="500px", key=c_id)
                    else:
                        st.error("图表数据已过期或丢失")

                st.session_state.messages.append({"role": "assistant", "content": response["output"]})

if __name__ == "__main__":
    main()