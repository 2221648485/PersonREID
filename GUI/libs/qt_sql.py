import pymysql
from REID.logger.log import get_logger

log = get_logger(__name__)


class MySQLDatabase:
    def __init__(self, host='localhost', user='root', password='123456', charset='utf8mb4'):
        self.host = host
        self.user = user
        self.password = password
        self.charset = charset
        self.connection = None

    def connect(self, database=None):
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=database,
                charset=self.charset,
                cursorclass=pymysql.cursors.DictCursor
            )
            return True
        except pymysql.Error as e:
            log.error(f"Error connecting to database: {e}")
            return False

    def create_database(self, db_name):
        if self.connect():
            with self.connection.cursor() as cursor:
                cursor.execute(f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{db_name}'")
                result = cursor.fetchone()
                if not result:
                    cursor.execute(f"CREATE DATABASE {db_name}")
                    log.info(f"Database {db_name} created successfully.")
            self.connection.close()
            return True
        return False

    def create_table(self, db_name):
        if self.connect(db_name):
            with self.connection.cursor() as cursor:
                sql = (f"CREATE TABLE IF NOT EXISTS {db_name} "
                       f"(id INT PRIMARY KEY AUTO_INCREMENT, name TEXT, category VARCHAR(255), box TEXT, feat TEXT, image VARCHAR(255))")
                cursor.execute(sql)
            self.connection.commit()
            self.connection.close()
            return True
        return False

    def init_db(self, db_name):
        if self.create_database(db_name):
            return self.create_table(db_name)
        return False

    def load_sql_feat_info(self, db_name):
        if self.connect(db_name):
            try:
                with self.connection.cursor() as cursor:
                    sql = f"SELECT name, feat FROM {db_name}"
                    cursor.execute(sql)
                    results = cursor.fetchall()
                    feat_list = []
                    label_list = []
                    for row in results:
                        label_list.append(row['name'])
                        feat_list.append(list(map(float, row['feat'].split(','))))
                return feat_list, label_list
            except pymysql.Error as e:
                log.error(f"{db_name} query feature error: {e}")
            finally:
                self.connection.close()
        return [], []

    def add_register(self, db_name, name, category, box, feat, image):
        if self.connect(db_name):
            try:
                with self.connection.cursor() as cursor:
                    sql = f"INSERT INTO {db_name} (name, category, box, feat, image) VALUES (%s, %s, %s, %s, %s)"
                    cursor.execute(sql, (name, category, box, feat, image))
                self.connection.commit()
                print("Query executed successfully")
                return True
            except pymysql.Error as e:
                print(f"Error: {e}")
                log.error(f"{db_name} insert data error: {e}")
            finally:
                self.connection.close()
        return False


if __name__ == "__main__":
    db = MySQLDatabase()
    print(db.init_db("reid"))
