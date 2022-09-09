import os

def test_and_commit(): #pragma: no cover
    """
    run all tests (testing.test_all) and record line coverage
    
    a commit made by this script should be included in all
    pull or merge requests near the tip of the latest changes
    """

    test_all_command = "coverage run -m sepia.test.test_all"
    write_report_command = "coverage report > coverage.txt"

    os.system(test_all_command)
    os.system(write_report_command)

    with open("coverage.txt", "r") as f:
        for my_line in f.readlines():
            if "TOTAL" in my_line:
                total_coverage = my_line

    add_coverage_command = "git add coverage.txt"
    commit_command = f"git commit -m 'test commit coverage: {total_coverage}'"

    os.system(add_coverage_command)
    os.system(commit_command)

if __name__ == "__main__": #pragma: no cover
    test_and_commit()
