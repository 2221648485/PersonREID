from PySide6.QtSql import QSqlDatabase, QSqlQuery
from REID.logger.log import get_logger
import REID.config.model_cfgs as cfgs

log_info = get_logger(__name__)

# 创建数据库
def init_db(db_path, db_name):
    try:
        db = QSqlDatabase.addDatabase("QSQLITE")
        db.setDatabaseName(db_path)
        res = db.open()
        if not db.open():
            error_msg = db.lastError().text()
            print(f"Error: Unable to open database - {error_msg}")
            log_info.error(f"{db_path}_{db_name} Unable to open database: {error_msg}")
            return False
        query = QSqlQuery()
        query.exec(
            f"CREATE TABLE IF NOT EXISTS {db_name} (id integer primary key, name text, category varchar, box text, feat text, image varchar)")
        return True
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        log_info.error(f"{db_path}_{db_name} An unexpected error occurred: {e}")
        return False


def check(func, *args):
    if not func(*args):
        raise ValueError(func.__self__.lastError())


def load_sql_feat_info(db_path, db_name):
    db = QSqlDatabase.addDatabase("QSQLITE")
    db.setDatabaseName(db_path)
    if not db.open():
        print("Error: Unable to open database")
        log_info.error("{}_{} Unable to open database.".format(db_path, db_name))
        return False
    query = QSqlQuery()
    query.exec("SELECT name, feat FROM {}".format(db_name))
    feat_list = []
    label_list = []
    if not query.isActive():
        log_info.error("{}_{} query feature error.".format(db_path, db_name))
        print("Error:", query.lastError().text())
    else:
        results = []
        while query.next():
            # 通过列索引检索数据
            data1 = query.value(0)  # 第一列的值
            data2 = query.value(1)  # 第二列的值
            label_list.append(data1)
            feat_list.append(list(map(float, data2.split(','))))
    return feat_list, label_list


def _add_register(db_path, db_name, name, category, box, feat, image):
    db = QSqlDatabase.addDatabase("QSQLITE")
    db.setDatabaseName(db_path)
    if not db.open():
        print("Error: Unable to open database")
        log_info.error("{}_{} Unable to open database.".format(db_path, db_name))
        return False
    q = QSqlQuery()
    INSERT_BOOK_SQL = "insert into {}(name, category, box, feat, image) values(?, ?, ?, ?, ?)".format(db_name)
    check(q.prepare, INSERT_BOOK_SQL)
    q.addBindValue(name)
    q.addBindValue(category)
    q.addBindValue(box)
    q.addBindValue(feat)
    q.addBindValue(image)
    q.exec()
    if q.lastError().isValid():
        # 查询执行失败，输出错误信息
        print("Error:", q.lastError().text())
    else:
        # 查询执行成功
        print("Query executed successfully")


if __name__ == "__main__":
    init_db(cfgs.DB_PATH, cfgs.DB_NAME)
