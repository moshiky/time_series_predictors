
import time
import os
import matplotlib.pylab as plt
from datetime import datetime, timedelta
import csv


class CourseraParser:

    # folders
    OUTPUT_DIR_PATH = r'output'
    GRAPHS_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, 'graphs')

    # normalized course graph size
    NORMALIZED_COURSE_GRAPH_LENGTH = 10
    MAX_NORMALIZED_VALUE = 1.0
    NORMALIZED_VALUE_INTERVAL = 0.05

    # date format
    DATE_FORMAT = '%Y-%m-%d'

    # Posts csv consts
    POSTS_CSV_COURSE_ID_COLUMN_INDEX = 2
    POSTS_CSV_USER_ID_COLUMN_INDEX = 5
    POSTS_CSV_USER_TYPE_COLUMN_INDEX = 6
    POSTS_CSV_TIMESTAMP_COLUMN_INDEX = 7
    POSTS_CSV_NORMALIZED_RELATIVE_POSTING_TIME_COLUMN_INDEX = 8

    # Course information csv consts
    COURSE_INFO_CSV_COURSE_ID_COLUMN_INDEX = 1
    COURSE_INFO_CSV_WEEKS_COLUMN_INDEX = 2
    COURSE_INFO_CSV_START_DATE_COLUMN_INDEX = 4
    COURSE_INFO_CSV_END_DATE_COLUMN_INDEX = 5

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
    def produce_course_duration_log(post_records, per_week=False, course_info=None):

        if not per_week:
            number_of_time_pieces = int(CourseraParser.MAX_NORMALIZED_VALUE / CourseraParser.NORMALIZED_VALUE_INTERVAL) + 1
            user_activity = dict()
            posts_per_timepiece = [0] * number_of_time_pieces

            for record in post_records:
                user_id = int(record[CourseraParser.POSTS_CSV_USER_ID_COLUMN_INDEX])
                user_type = record[CourseraParser.POSTS_CSV_USER_TYPE_COLUMN_INDEX]
                relative_posting_time = \
                    float(record[CourseraParser.POSTS_CSV_NORMALIZED_RELATIVE_POSTING_TIME_COLUMN_INDEX])

                # skip irrelevant records
                if user_id == 0 or user_type != 'Student' \
                        or CourseraParser.MAX_NORMALIZED_VALUE < relative_posting_time or relative_posting_time < 0:
                    # irrelevant record
                    continue

                timepiece_index = int(relative_posting_time / CourseraParser.NORMALIZED_VALUE_INTERVAL)
                posts_per_timepiece[timepiece_index] += 1

                if user_id not in user_activity.keys():
                    user_activity[user_id] = [0] * number_of_time_pieces
                user_activity[user_id][timepiece_index] += 1

            return posts_per_timepiece, user_activity

        else:
            extra_activity_weeks = 2
            posts_per_week = [0] * (course_info[1] + extra_activity_weeks + 1)
            course_start_date = datetime.strptime(course_info[0], "%m/%d/%Y")
            last_valid_activity_date = course_start_date + timedelta(days=(len(posts_per_week)*7))
            for post_record in post_records:
                post_date = \
                    datetime.fromtimestamp(
                        int(post_record[CourseraParser.POSTS_CSV_TIMESTAMP_COLUMN_INDEX])
                    )
                if post_date < course_start_date or post_date > last_valid_activity_date:
                    continue
                post_week_index = int((post_date - course_start_date).days / 7)
                posts_per_week[post_week_index] += 1

            return posts_per_week

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
        normalized_course_graphs = list()
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
            # columns = ['user_id'] + sorted(list(padded_records[list(padded_records.keys())[0]].keys()))
            # CourseraParser.save_as_csv(
            #     r'output/course_{course_id}.csv'.format(course_id=course_id), records_for_csv, columns
            # )

            normalized_course_avg_graph = CourseraParser.get_normalized_course_graph(records_for_csv)
            normalized_course_graphs.append([course_id] + normalized_course_avg_graph)

        # save normalized graphs
        columns = \
            ['course_id'] \
            + ['per.{per_id}'.format(per_id=i) for i in range(CourseraParser.NORMALIZED_COURSE_GRAPH_LENGTH)]

        CourseraParser.save_as_csv(
            r'output/normalized_course_graphs.csv', normalized_course_graphs, columns
        )

        print('done')

    @staticmethod
    def get_timepiece_headers():
        number_of_time_pieces = int(CourseraParser.MAX_NORMALIZED_VALUE / CourseraParser.NORMALIZED_VALUE_INTERVAL) + 1
        headers = []
        for i in range(number_of_time_pieces):
            headers.append(str((i+1) * CourseraParser.NORMALIZED_VALUE_INTERVAL))
        return headers

    def parse_posts_course_duration(self, posts_csv_file_path, courses_csv_file_path):
        # read csv files
        print('reading posts input file..')
        post_records = self.parse_posts_file(posts_csv_file_path)
        print('found {num_posts} posts'.format(num_posts=len(post_records)))

        print('reading course details file..')
        course_records = self.parse_posts_file(courses_csv_file_path)
        print('found {num_courses} course info records'.format(num_courses=len(course_records)))

        # fetch course information
        course_info = dict()
        for course_info_record in course_records:
            course_info[course_info_record[CourseraParser.COURSE_INFO_CSV_COURSE_ID_COLUMN_INDEX]] = [
                course_info_record[CourseraParser.COURSE_INFO_CSV_START_DATE_COLUMN_INDEX],
                int(course_info_record[CourseraParser.COURSE_INFO_CSV_WEEKS_COLUMN_INDEX])
            ]

        # split records by courses
        print('splitting posts by course id..')
        course_posts = dict()
        for record in post_records:
            course_id = record[CourseraParser.POSTS_CSV_COURSE_ID_COLUMN_INDEX]
            if course_id in course_posts.keys():
                course_posts[course_id].append(list(record))
            else:
                course_posts[course_id] = [list(record)]
        print('found {num_courses} courses'.format(num_courses=len(course_posts.keys())))

        # create csv headers
        timepiece_headers = CourseraParser.get_timepiece_headers()
        csv_headers = ['course_id'] + timepiece_headers

        # produce csv for each course
        csv_lines = list()
        user_total_activity = dict()
        # os.mkdir(CourseraParser.GRAPHS_DIR_PATH)
        course_num_users = dict()
        for course_id in course_posts.keys():
            print('parsing course id: {course_id}'.format(course_id=course_id))
            # call to produce_course_log_for_student_csv
            # total_posts_by_relative_timepiece, user_activity = \
            #     CourseraParser.produce_course_duration_log(course_posts[course_id])
            posts_per_week = \
                CourseraParser.produce_course_duration_log(
                    course_posts[course_id],
                    True,
                    course_info[course_id]
                )

            # store in csv with course id
            csv_row = [course_id] + posts_per_week
            csv_lines.append(csv_row)

            # save course chart
            # file_name = 'posts_over_time_course_{course_id}.png'.format(course_id=course_id)
            # CourseraParser.save_chart(
            #     os.path.join(CourseraParser.GRAPHS_DIR_PATH, file_name),
            #     timepiece_headers,
            #     total_posts_by_relative_timepiece
            # )

            # save user charts
            # course_folder_path = os.path.join(CourseraParser.GRAPHS_DIR_PATH, course_id)
            # os.mkdir(course_folder_path)
            # for user_id in user_activity.keys():
            #     file_name = 'course_{course_id}_user_{user_id}.png'.format(course_id=course_id, user_id=user_id)
            #     file_path = os.path.join(course_folder_path, file_name)
            #     CourseraParser.save_chart(file_path, timepiece_headers, user_activity[user_id])

            # add to total activity counters
            # for user_id in user_activity.keys():
            #     if user_id not in user_total_activity.keys():
            #         user_total_activity[user_id] = [user_activity[user_id], 1]
            #     else:
            #         user_total_activity[user_id][1] += 1
            #         for i in range(len(user_activity[user_id])):
            #             user_total_activity[user_id][0][i] += user_activity[user_id][i]

            # get number of users
            # course_num_users[course_id] = len(user_activity.keys())

        # print('number of users: {num_users}'.format(num_users=len(user_total_activity.keys())))
        # course_counters = [0] * 20
        # for user_id in user_total_activity.keys():
        #     course_counters[user_total_activity[user_id][1]] += 1
        #     # if user_total_activity[user_id][1] > 100:
        #     #     file_name = \
        #     #         'courses_{num_courses}_user_{user_id}.png'.format(
        #     #             num_courses=user_total_activity[user_id][1], user_id=user_id
        #     #         )
        #     #     file_path = os.path.join(CourseraParser.GRAPHS_DIR_PATH, file_name)
        #     #     CourseraParser.save_chart(file_path, timepiece_headers, user_total_activity[user_id][0])
        #
        # for i in range(len(course_counters)):
        #     print('{num_courses} : {num_users}'.format(num_courses=i, num_users=course_counters[i]))

        CourseraParser.save_as_csv(
            r'output/posts_per_course_week.csv', csv_lines
        )

        # print('course_id:num_users')
        # for course_id in course_num_users.keys():
        #     print('{course_id}:{num_users}'.format(course_id=course_id, num_users=course_num_users[course_id]))

        print('done')

    @staticmethod
    def save_chart(file_path, x_values, y_values, start_v_lines=None, end_v_lines=None):
        plt.clf()
        plt.plot(x_values, y_values)
        # if start_v_lines is not None:
        #     for x_value in start_v_lines:
        #         plt.axvline(x=x_value, color='red')
        # if end_v_lines is not None:
        #     for x_value in end_v_lines:
        #         plt.axvline(x=x_value, color='blue')
        plt.savefig(fname=file_path)
        plt.clf()

    @staticmethod
    def get_normalized_course_graph(course_records):
        # calculate avg. graph
        dates_count = len(course_records[0]) - 1
        avg_graph = [0] * dates_count
        for row in course_records:
            for i in range(1, dates_count+1):
                avg_graph[i-1] += row[i]

        for i in range(len(avg_graph)):
            avg_graph[i] /= float(len(course_records))

        # normalize avg. graph
        org_parts_per_normalized_part = int(dates_count / CourseraParser.NORMALIZED_COURSE_GRAPH_LENGTH)
        normalized_avg_graph = [0] * CourseraParser.NORMALIZED_COURSE_GRAPH_LENGTH
        for i in range(CourseraParser.NORMALIZED_COURSE_GRAPH_LENGTH):
            start_index = i * org_parts_per_normalized_part
            end_index = min((i+1) * org_parts_per_normalized_part, len(avg_graph))
            actual_parts_quantity = end_index - start_index

            normalized_avg_graph[i] = sum(avg_graph[start_index:end_index]) / float(actual_parts_quantity)

        # return normalized graph
        return normalized_avg_graph

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
    def save_as_csv(output_csv_file_path, records, columns=None):
        with open(output_csv_file_path, 'at', newline='') as output_csv_file:
            writer = csv.writer(output_csv_file)
            try:
                if columns is not None:
                    writer.writerow(columns)
                writer.writerows(records)
            except Exception as ex:
                print('ERROR: Failed writing csv file. exception: ' + str(ex))
                raise

    @staticmethod
    def get_weeks_since_date(start_date, event_date):
        days_since_start = (datetime.fromtimestamp(int(event_date)) - datetime.fromtimestamp(int(start_date))).days
        return int(days_since_start / 7)

    def parse_posts_for_users(self, posts_csv_file_path, courses_csv_file_path):
        # read csv files
        print('reading posts input file..')
        post_records = self.parse_posts_file(posts_csv_file_path)
        print('found {num_posts} posts'.format(num_posts=len(post_records)))

        print('reading course details file..')
        course_records = self.parse_posts_file(courses_csv_file_path)
        print('found {num_courses} course info records'.format(num_courses=len(course_records)))

        # ## summarize users activity
        # gather post dates by user id
        users_activity = dict()
        for record in post_records:
            user_id = int(record[CourseraParser.POSTS_CSV_USER_ID_COLUMN_INDEX])
            user_type = record[CourseraParser.POSTS_CSV_USER_TYPE_COLUMN_INDEX]

            # skip irrelevant records
            if user_id == 0 or user_type != 'Student':
                # irrelevant record
                continue

            if user_id not in users_activity.keys():
                users_activity[user_id] = list()
            users_activity[user_id].append(list(record))

        os.mkdir(CourseraParser.GRAPHS_DIR_PATH)
        # for each user, print graph by week
        for user_id in users_activity.keys():
            # count num of courses
            user_courses = list()
            for user_post in users_activity[user_id]:
                if user_post[CourseraParser.POSTS_CSV_COURSE_ID_COLUMN_INDEX] not in user_courses:
                    user_courses.append(user_post[CourseraParser.POSTS_CSV_COURSE_ID_COLUMN_INDEX])

            if len(user_courses) < 5:
                continue

            # calculate number of activity weeks
            # > find start date and end date
            min_date = None
            max_date = None
            for user_post in users_activity[user_id]:
                post_date = user_post[CourseraParser.POSTS_CSV_TIMESTAMP_COLUMN_INDEX]
                if min_date is None or post_date < str(min_date):
                    min_date = post_date
                if max_date is None or post_date > str(max_date):
                    max_date = post_date
            # > calculate diff
            num_weeks = CourseraParser.get_weeks_since_date(min_date, max_date) + 1

            # for each post date - calculate in which week it is
            week_counters = [0] * num_weeks
            for user_post in users_activity[user_id]:
                post_date = user_post[CourseraParser.POSTS_CSV_TIMESTAMP_COLUMN_INDEX]
                week_id = CourseraParser.get_weeks_since_date(min_date, post_date)
                week_counters[week_id] += 1

            # find out when courses start - in which week
            start_weeks = list()
            end_weeks = list()
            for course_id in user_courses:
                course_start_date = None
                course_weeks = 0
                for course_record in course_records:
                    if course_id == course_record[CourseraParser.COURSE_INFO_CSV_COURSE_ID_COLUMN_INDEX]:
                        course_start_date = course_record[CourseraParser.COURSE_INFO_CSV_START_DATE_COLUMN_INDEX]
                        course_weeks = int(course_record[CourseraParser.COURSE_INFO_CSV_WEEKS_COLUMN_INDEX])
                        break

                # convert dates to timestamps
                course_start_date = time.mktime(datetime.strptime(course_start_date, "%m/%d/%Y").timetuple())

                start_week_index = CourseraParser.get_weeks_since_date(min_date, course_start_date)
                if start_week_index not in start_weeks:
                    start_weeks.append(start_week_index)
                end_week_index = start_week_index + course_weeks
                if end_week_index not in end_weeks:
                    end_weeks.append(end_week_index)

            # save user chart
            file_path = \
                os.path.join(
                    CourseraParser.GRAPHS_DIR_PATH,
                    'user_{user_id}_courses_{num_courses}.png'.format(user_id=user_id, num_courses=len(user_courses))
                )
            CourseraParser.save_chart(file_path, list(range(num_weeks)), week_counters, start_weeks, end_weeks)

if __name__ == '__main__':

    BASE_FOLDER = r'datasets/courseraforums-master/data'
    POSTS_FILE_NAME = 'course_posts.csv'
    COURSE_INFORMATION_FILE_NAME = 'course_information.csv'

    parser = CourseraParser()
    parser.parse_posts_course_duration(
        os.path.join(BASE_FOLDER, POSTS_FILE_NAME), os.path.join(BASE_FOLDER, COURSE_INFORMATION_FILE_NAME)
    )
    # parser.parse_posts_for_users(
    #     os.path.join(BASE_FOLDER, POSTS_FILE_NAME),
    #     os.path.join(BASE_FOLDER, COURSE_INFORMATION_FILE_NAME)
    # )


"""
    >> 1. add script that takes the average posts per day from each course and makes a single csv from it
    >> 2. normalize time periods 
        >> calculate avg. posts at each time period
    >>  create one csv 
        
    >> consider course begin and end dates
"""