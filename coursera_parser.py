
from datetime import datetime, timedelta
import csv


class CourseraParser:

    # date format
    DATE_FORMAT = '%Y-%m-%d'

    # Posts csv consts
    POSTS_CSV_COURSE_ID_COLUMN_INDEX = 2
    POSTS_CSV_USER_ID_COLUMN_INDEX = 5
    POSTS_CSV_USER_TYPE_COLUMN_INDEX = 6
    POSTS_CSV_TIMESTAMP_COLUMN_INDEX = 7

    def __init__(self):
        pass

    @staticmethod
    def parse_posts_file(csv_file_path):
        # read file
        with open(csv_file_path, 'rt') as file_handle:
            reader = csv.reader(file_handle)
            file_lines = list(reader)

        # return records
        return file_lines[1:]

    @staticmethod
    def produce_course_log_for_student_csv(post_records):
        """
        receives only records of post of specific course, or of all courses
        :param post_records:
        :return:
        """

        # split by user
        # get all of its posts
        # translate date to day
        # count how much for each day of posting
        # add to list a line that says that this user post X post on date Y, Z posts on date K and so on
        # caller should produce the output csv that associate with the course and orders by date
        posts_by_user = dict()
        # log time-line limits
        min_date = None
        max_date = None
        for record in post_records:
            user_id = int(record[CourseraParser.POSTS_CSV_USER_ID_COLUMN_INDEX])
            user_type = record[CourseraParser.POSTS_CSV_USER_TYPE_COLUMN_INDEX]

            # skip irrelevant records
            if user_id == 0 or user_type != 'Student':
                # irrelevant record
                continue

            # transform record
            transformed_record = list(record)
            post_timestamp = int(transformed_record[CourseraParser.POSTS_CSV_TIMESTAMP_COLUMN_INDEX])
            post_date = datetime.fromtimestamp(post_timestamp).strftime(CourseraParser.DATE_FORMAT)

            # log time-line limits
            if min_date is None or post_date < str(min_date):
                min_date = post_date
            if max_date is None or post_date > str(max_date):
                max_date = post_date

            # add post to user counters
            if user_id in posts_by_user.keys():
                if post_date in posts_by_user[user_id].keys():
                    posts_by_user[user_id][post_date] += 1
                else:
                    posts_by_user[user_id][post_date] = 1
            else:
                posts_by_user[user_id] = {post_date: 1}

        # order each list by the date
        posts_by_user = {user_id: dict(sorted(posts_by_user[user_id].items())) for user_id in posts_by_user.keys()}

        return posts_by_user, min_date, max_date

    def parse_posts(self, csv_file_path):
        # read csv file
        print('reading input file..')
        file_records = self.parse_posts_file(csv_file_path)
        print('found {num_posts} posts'.format(num_posts=len(file_records)))

        # split records by courses
        print('splitting posts by course id..')
        course_posts = dict()
        for record in file_records:
            course_id = record[CourseraParser.POSTS_CSV_COURSE_ID_COLUMN_INDEX]
            if course_id in course_posts.keys():
                course_posts[course_id].append(list(record))
            else:
                course_posts[course_id] = [list(record)]
        print('found {num_courses} courses'.format(num_courses=len(course_posts.keys())))

        # produce csv for each course
        for course_id in course_posts.keys():
            print('parsing course id: {course_id}'.format(course_id=course_id))
            # call to produce_course_log_for_student_csv
            posts_by_user, min_date, max_date = CourseraParser.produce_course_log_for_student_csv(course_posts[course_id])

            # pad records according to max and min dates
            padded_records = CourseraParser.pad_records(posts_by_user, min_date, max_date)

            # store in csv with course id
            records_for_csv = [
                [record[0]]
                + [record[1][date_key] for date_key in sorted(record[1].keys())]
                for record in padded_records.items()
            ]
            columns = ['user_id'] + sorted(list(padded_records[list(padded_records.keys())[0]].keys()))
            CourseraParser.save_as_csv(
                r'output/course_{course_id}.csv'.format(course_id=course_id), records_for_csv, columns
            )

        print('done')

    @staticmethod
    def get_dates_between_dates(min_date, max_date):
        min_date_parts = [int(x) for x in min_date.split('-')]
        max_date_parts = [int(x) for x in max_date.split('-')]

        min_date_dt = datetime(min_date_parts[0], min_date_parts[1], min_date_parts[2])
        max_date_dt = datetime(max_date_parts[0], max_date_parts[1], max_date_parts[2])

        delta_days = (max_date_dt - min_date_dt).days

        all_days_list = list()
        for i in range(delta_days + 1):
            all_days_list.append((min_date_dt + timedelta(days=i)).strftime(CourseraParser.DATE_FORMAT))

        return all_days_list

    @staticmethod
    def pad_records(posts_by_user, min_date, max_date):
        all_dates = CourseraParser.get_dates_between_dates(min_date, max_date)

        padded_dict = dict()
        for user_id in posts_by_user.keys():
            user_dates_dict = dict()
            for date_string in all_dates:
                if date_string in posts_by_user[user_id].keys():
                    user_dates_dict[date_string] = posts_by_user[user_id][date_string]
                else:
                    user_dates_dict[date_string] = 0

            padded_dict[user_id] = user_dates_dict

        return padded_dict

    @staticmethod
    def save_as_csv(output_csv_file_path, records, columns):
        with open(output_csv_file_path, 'at', newline='') as output_csv_file:
            writer = csv.writer(output_csv_file)
            try:
                writer.writerow(columns)
                writer.writerows(records)
            except Exception as ex:
                print('ERROR: Failed writing csv file. exception: ' + str(ex))
                raise

if __name__ == '__main__':
    parser = CourseraParser()
    parser.parse_posts(r'datasets/courseraforums-master/data/course_posts.csv')
