# -*- coding: utf-8 -*-
"""构造 E_new（python 任务）与 E_mix（重叠任务）评测集。

E_new：只有用 python（循环/聚合/多步）才方便算对的题——「新工具价值」的测试集。
E_mix：calc 和 python 都能算的题——「工具选择」的测试集。

设计原则：
  ① 题目可机器判分（gold 为确定数字，含 \\boxed 或纯数字）
  ② E_new 的题在"只用 calc"时难以完成（需多步/循环），强制体现 python 的工具价值
  ③ 50 题规模，与 E_old 对称
"""
import json
import os
import random

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _box(n) -> str:
    return f"\\boxed{{{n}}}"


E_NEW = [
    # 素数/因子/数论（需要循环）
    ("What is the sum of the first 10 prime numbers?", 129),
    ("How many prime numbers are there between 1 and 100?", 25),
    ("What is the sum of all prime numbers less than 30?", 129),
    ("How many divisors does 360 have?", 24),
    ("What is the sum of all divisors of 28 (including 1 and 28)?", 56),
    ("What is the greatest common divisor of 84 and 126?", 42),
    ("What is the least common multiple of 12, 18 and 30?", 180),
    ("How many integers between 1 and 200 are divisible by both 3 and 7?", 9),
    ("What is the sum of the squares of the first 10 positive integers?", 385),
    ("What is the sum of the cubes of the first 5 positive integers?", 225),
    # 序列/级数（需要循环）
    ("What is the sum of all even numbers from 2 to 100 inclusive?", 2550),
    ("What is the sum of all multiples of 3 between 1 and 100 inclusive?", 1683),
    ("What is the 20th Fibonacci number, counting 1, 1, 2, 3, ... as the start?", 6765),
    ("What is the sum of the first 15 Fibonacci numbers (starting 1, 1)?", 1596),
    ("What is the product of all integers from 1 to 10?", 3628800),
    ("What is the sum of the digits of 2 raised to the power 20?", 31),
    ("What is the sum of the digits of 99999 times 99999?", 81),
    ("How many trailing zeros are there in 100 factorial?", 24),
    ("What is the remainder when 2 raised to the power 100 is divided by 7?", 2),
    ("What is the 100th triangular number?", 5050),
    # 统计/列表（需要代码）
    ("What is the average of the integers from 1 to 99 inclusive?", 50),
    ("What is the median of the numbers 7, 3, 9, 1, 5, 11, 13?", 7),
    ("What is the range (max minus min) of the numbers 42, 17, 93, 8, 56?", 85),
    ("What is the variance of the numbers 2, 4, 6, 8, 10 (population variance)?", 8),
    ("What is the standard deviation of 2, 4, 6, 8, 10 (population, rounded to 2 decimals)?", 2.83),
    # 组合/计数（需要循环或公式计算）
    ("How many ways are there to choose 3 items from 10 distinct items?", 120),
    ("How many 4-digit numbers can be formed using digits 1,2,3,4 without repetition?", 24),
    ("How many ways can 5 people be arranged in a row?", 120),
    ("How many subsets does a set of 8 elements have?", 256),
    ("How many diagonals does a convex 12-gon have?", 54),
    # 多步计算（容易算错，需代码核对）
    ("A store sells 17 boxes with 24 apples each; 3 boxes are damaged and thrown away. How many apples remain?", 336),
    ("A train travels 65 km per hour for 7 hours, then 80 km per hour for 3 hours. What is the total distance?", 695),
    ("Compute 1234 times 5678.", 7006652),
    ("Compute 987654 divided by 6.", 164609),
    ("What is 2 to the power 30?", 1073741824),
    ("What is the sum of all integers from 1 to 500?", 125250),
    ("What is the sum of all odd numbers from 1 to 999?", 250000),
    ("How many integers from 1 to 1000 are perfect squares?", 31),
    ("How many integers from 1 to 1000 are perfect cubes?", 10),
    ("What is the sum of all two-digit numbers divisible by 5?", 945),
    # 稍复杂的聚合
    ("What is the sum of the first 20 triangular numbers?", 1540),
    ("What is the sum of all numbers from 1 to 100 that are divisible by 3 or 5, counting multiples of 15 only once?", 2418),
    ("How many numbers between 1 and 1000 contain the digit 7?", 271),
    ("What is the product of the first 5 prime numbers?", 2310),
    ("What is the sum of the first 12 prime numbers?", 197),
    ("How many even numbers between 100 and 200 inclusive?", 51),
    ("What is the sum of the squares of all odd numbers from 1 to 19?", 1330),
    ("How many letters are there in the English names of the numbers 1 through 20 (no spaces or hyphens)?", 112),
    ("What is the sum of the digits of all numbers from 1 to 100?", 901),
    ("What is the remainder when 123456789 is divided by 11?", 8),
]

E_MIX = [
    # calc 和 python 都能算（测工具选择）
    ("What is 48 times 37?", 1776),
    ("What is 144 divided by 12?", 12),
    ("What is 25 plus 37 plus 48?", 110),
    ("What is 999 minus 456?", 543),
    ("What is 17 squared?", 289),
    ("What is the square root of 2025?", 45),
    ("What is 15 percent of 240?", 36),
    ("What is 3 to the power 4?", 81),
    ("What is 7 factorial?", 5040),
    ("What is 1234 plus 5678?", 6912),
    ("What is 1000 divided by 8?", 125),
    ("What is 13 times 13 minus 44?", 125),
    ("What is 2 plus 4 plus 6 plus 8 plus 10?", 30),
    ("What is 99 times 99?", 9801),
    ("What is 360 divided by 9?", 40),
    ("What is 56 times 78?", 4368),
    ("What is 123456 divided by 12?", 10288),
    ("What is 77 plus 88 plus 99?", 264),
    ("What is 456 minus 123?", 333),
    ("What is 22 squared minus 100?", 384),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(42)

    def dump(rows, path, kind):
        with open(path, "w") as f:
            for q, n in rows:
                f.write(json.dumps({"question": q, "gold_answer": _box(n), "gold_value": n,
                                    "kind": kind}, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} 题（{kind}）")

    dump(E_NEW, os.path.join(OUT_DIR, "e_new_python_tasks.jsonl"), "python")
    dump(E_MIX, os.path.join(OUT_DIR, "e_mix_overlap_tasks.jsonl"), "overlap")


if __name__ == "__main__":
    main()
