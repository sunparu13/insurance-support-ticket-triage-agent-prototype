import argparse
import csv
import os
import random
import json
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter


class Evaluation:

    def failure_cases(self, result: dict) -> bool:
            return result.get("status") == "error" 

    #stats
    def result_summary(self, results):
        """
        get the summary of the triage results, including status counts, status rates, error types and error stages
        """
        total = len(results)
        counts = Counter(item.get("status") or "Unknown" for item in results)
        for status in ("ok", "needs_more_info", "needs_review", "error"):
            if status not in counts:
                counts[status] = 0
        failures = [item for item in results if self.failure_cases(item)]
        return {
            "status_counts": dict(counts),
            "status_rates": {key: count / total if total else 0.0 for key, count in counts.items()},
            "error_types": dict(Counter(item.get("error_type") or "Unknown" for item in failures)),
            "error_stages": dict(Counter(item.get("error_stage") or "Unknown" for item in failures)),
        }

    def ratio_tech_failure(self, results):
        """
        calculate the ratio of technical failures in the results
        """
        total = len(results)
        tech_failures = sum(self.failure_cases(result) for result in results)
        return tech_failures / total if total > 0 else 0

    def urgency_accuracy_pr_f1(self, ground_truth, predicted):
        """
        evaluate the urgency independently of downstream failures.
        notice: missing or invalid urgency predictions count as incorrect, not excluded.
        """
        labels = ["Low", "Medium", "High"]
        ground_truth_dict = {item['ticket_id']: item for item in ground_truth}
        true_labels = [ground_truth_dict[item['ticket_id']]['priority'].strip().title() for item in predicted]
        pred_labels = []
        for item in predicted:
            urgency = item.get("urgency")
            pred_labels.append(urgency if urgency in labels else "Unknown")

        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, labels=labels, average='macro', zero_division=0)
        recall = recall_score(true_labels, pred_labels, labels=labels, average='macro', zero_division=0)
        macro_f1 = f1_score(true_labels, pred_labels, labels=labels, average='macro', zero_division=0)
        high_recall = recall_score(true_labels, pred_labels, labels=["High"], average='macro', zero_division=0) if "High" in true_labels else None
        return {"sample_count": len(true_labels), "accuracy": accuracy,
                "macro_precision": precision, "macro_recall": recall,
                "macro_f1": macro_f1,
                "high_count": true_labels.count("High"),
                "high_recall": high_recall}

    def queue_action_agreement(self, ground_truth, predicted):
        """compare next_action with mapped queues"""
        queue_to_action = {
            "Billing and Payments": "Forward to billing team",
            "Technical Support": "Forward to technical support",
            "IT Support": "Forward to technical support",
        }

        source = {item["ticket_id"]: item for item in ground_truth}
        true_labels = []
        pred_labels = []

        for result in predicted:
            original = source[result["ticket_id"]]
            queue = original["queue"].strip()
            if queue not in queue_to_action:
                continue

            true_labels.append(queue_to_action[queue])
            pred_labels.append(result.get("next_action") or "Unknown")

        if not true_labels:
            print("No tickets have a mapped queue")
            return {"sample_count": 0, "accuracy": None,
                    "macro_precision": None, "macro_recall": None, "macro_f1": None,
                    "queue_to_action": queue_to_action}

        labels = sorted(set(queue_to_action.values()))
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, labels=labels, average="macro", zero_division=0)
        recall = recall_score(true_labels, pred_labels, labels=labels, average="macro", zero_division=0)
        macro_f1 = f1_score(true_labels, pred_labels, labels=labels, average="macro", zero_division=0)
            
        return {"sample_count": len(true_labels), "accuracy": accuracy,
                "macro_precision": precision, "macro_recall": recall,
                "macro_f1": macro_f1, "queue_to_action": queue_to_action}


    def human_evaluation(self, 
                         ground_truth, 
                         predicted, 
                         review_csv,
                         num_initial_eval = 10, 
                         seed = 42, 
                         pass_score = 4): #unfortunately, this part is over time budget 12 hours.
        """
        human provides the subject evaluation on topic/urgency/action/note between ground truth and the agent result
        you can choose the llm-as-judge or real human review (expansion tbd)
        Notice: evaluation and following evaluation should always be based on the same result csv, especially human evaluation
        Args:
            ground_truth: list of dicts, each dict contains the original ticket information
            predicted: list of dicts, each dict contains the agent's triage result
            review_csv: str, path to the csv file for human review
            num_initial_eval: int, number of tickets to evaluate in the initial review
            seed: int, random seed for sampling tickets
            pass_score: int, minimum score required in every dimension to pass
        Returns:
            Dictionary containing review mode, status, sample count and scores.
        """
        scores = ["topic_eva", "urgency_eva", "action_eva", "notes_eva"]
        fields = ["ticket_id", "text", "topic", "urgency", "next_action", "status", "notes", *scores]
        reviews = []
        exists = os.path.exists(review_csv)

        if exists:
            with open(review_csv, "r", encoding="utf-8-sig", newline="") as f:
                reviews = list(csv.DictReader(f))
            ans = input("Do you want to append more tickets to be reviewed? (y/n): ").strip().lower()
            add_more = ans == "y"
        else:
            add_more = False #not exist, create a new one

        if add_more or not exists:
            if add_more:
                #check whether number of remaining tickets is enough
                existing_ids = {int(row["ticket_id"]) for row in reviews}
                remaining_tickets = [result for result in predicted if result["ticket_id"] not in existing_ids] #find tickets in predicted that are not being added into the review table yet
                if not remaining_tickets:
                    print("No remaining tickets to add")
                    selected = []
                else:
                    count = int(input(
                        f"How many additional tickets? (1 to {len(remaining_tickets)}): "
                    ).strip())
                    if not (1 <= count <= len(remaining_tickets)):
                        raise ValueError(f"Count must be between 1 and {len(remaining_tickets)}")
                    selected = random.Random(seed).sample(remaining_tickets, count)
            else: #initial review
                selected = random.Random(seed).sample(predicted, min(num_initial_eval, len(predicted)))
            
            #generate table
            source = {row["ticket_id"]: row for row in ground_truth} #notice the k:v pair
            csv_mode = "a" if exists else "w"
            os.makedirs(os.path.dirname(os.path.abspath(review_csv)), exist_ok=True)

            with open(review_csv, csv_mode, encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                for result in selected:
                    row = {key: result.get(key, "") for key in fields}  
                    original = source[result["ticket_id"]] #choose the original ticket from the ground truth based on ticket_id
                    row["text"] = (
                        f"Subject: {original.get('subject', '')}\n"
                        f"Body: {original.get('body', '')}"
                    )
                    writer.writerow(row)

            if selected:
                print(f"Saved {len(selected)} tickets to {review_csv}. Fill the four *_eva columns with 1 to 5.")
                return {"mode": "human", "status": "pending",
                        "sample_count": len(reviews) + len(selected),
                        "average_scores": None, "pass_rate": None}
        if not reviews:
            raise ValueError("The review table is empty")

        for row in reviews:
            for score in scores:
                value = (row.get(score) or "").strip()
                if value not in {"1", "2", "3", "4", "5"}:
                    print("You input wrong values or still are not completely filled out the *_eva columns. Complete every score with 1 to 5, then run again.")
                    return {"mode": "human", "status": "pending",
                            "sample_count": len(reviews),
                            "average_scores": None, "pass_rate": None} #just finish, modify by rerun
                row[score] = int(value)

        averages = {}
        for score in scores:
            average = sum(row[score] for row in reviews) / len(reviews)
            averages[score] = average
            print(f"Average {score}: {average:.2f}/5")

        pass_count = 0
        for row in reviews:
            passed = True
            for score in scores:
                if row[score] < pass_score:
                    passed = False
                    break
            if passed:
                pass_count += 1
        
        print(
            f"Review pass rate: {pass_count}/{len(reviews)} "
            f"({pass_count / len(reviews):.1%})"
        )

        return {"mode": "human", "status": "completed",
                "sample_count": len(reviews), "average_scores": averages,
                "pass_count": pass_count, "pass_rate": pass_count / len(reviews)}

    def save_evaluation(self, metrics, output_json):
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        history = []
        if os.path.exists(output_json):
            with open(output_json, encoding="utf-8") as f:
                history = json.load(f)
            # keep reports saved by the earlier single-report version.
            if isinstance(history, dict):
                history = [history]
        history.append(report)

        output_dir = os.path.dirname(os.path.abspath(output_json))
        os.makedirs(output_dir, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"Evaluation appended to {output_json} ({len(history)} runs)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", default="data/aa_dataset-tickets-multi-lang-5-2-50-version.csv",
                        help="Path to the original ticket csv")
    parser.add_argument("--result_json", required=True,
                        help="Path to the saved triage results")
    parser.add_argument("--human_review_csv", action="store_true",
                        help="Enable human evaluation (disabled by default)")
    parser.add_argument("--review_csv", default=None,
                        help="Human review table path (default: triage_human_review_<result filename suffix>.csv)")
    parser.add_argument("--output_json", default=None,
                        help="Evaluation history path (default: evaluation_results/evaluation_results_<result filename suffix>.json; each run appends a record)")
    parser.add_argument("--num_initial_eval", type=int, default=10,
                        help="Number of tickets in a new review table (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random sampling seed (default: 42)")
    parser.add_argument("--pass_score", type=int, choices=range(1, 6), default=4,
                        help="Minimum score required in every dimension to pass (default: 4)")
    
    args = parser.parse_args()

    result_name = os.path.splitext(os.path.basename(args.result_json))[0]
    result_suffix = result_name.removeprefix("triage_results_")

    if args.output_json is None:
        args.output_json = os.path.join("evaluation_results", f"evaluation_results_{result_suffix}.json")

    if args.review_csv is None:
        args.review_csv = f"triage_human_review_{result_suffix}.csv"


    if args.human_review_csv and args.num_initial_eval < 1:
        parser.error("Initial evaluation sample count must be positive.")

    #load the json result
    with open(args.result_json, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not results:
        raise ValueError(f"{args.result_json} is empty. Run main.py first.")

    #load the ground truth from the dataset
    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        ground_truth = []
        for ticket_id, row in enumerate(reader):
            row["ticket_id"] = ticket_id
            ground_truth.append(row)


#############start evaluation part#############################
    evaluator = Evaluation()
    #objective evaluation   
    #failure ratio
    failure_ratio = evaluator.ratio_tech_failure(results)
    print(f"Ticket total evaluation: {len(results)}, Technical failure ratio: {failure_ratio:.2%}")
    #urgency accuracy pr and f1
    urgency = evaluator.urgency_accuracy_pr_f1(ground_truth, results)
    print(f"Urgency: {urgency['sample_count']} tickets, Accuracy: {urgency['accuracy']:.2%}, "
          f"Macro precision: {urgency['macro_precision']:.2f}, "
          f"Macro recall: {urgency['macro_recall']:.2f}, Macro F1: {urgency['macro_f1']:.2f}",
          f"High urgency count: {urgency['high_count']}, High urgency recall: {urgency['high_recall']:.2%}" if urgency['high_count'] > 0 else "")

    #queue-action agreement on mapped queues
    queue_action = evaluator.queue_action_agreement(ground_truth, results)
    if queue_action["accuracy"] is not None:
        print(f"Queue-action agreement on mapped queues: {queue_action['sample_count']} tickets, Agreement: {queue_action['accuracy']:.2%}, "
              f"Macro precision: {queue_action['macro_precision']:.2f}, "
              f"Macro recall: {queue_action['macro_recall']:.2f}, Macro F1: {queue_action['macro_f1']:.2f}")
    
    #subjective evaluation from human review
    review = {"mode": "human", "status": "skipped"}
    if args.human_review_csv:  
        review = evaluator.human_evaluation(
            ground_truth, results,
            review_csv=args.review_csv,
            num_initial_eval=args.num_initial_eval,
            seed=args.seed,
            pass_score=args.pass_score,
        )

    evaluator.save_evaluation({
        "ticket_count": len(results),
        "technical_failure_rate": failure_ratio,
        'results_summary': evaluator.result_summary(results),
        "urgency": urgency,
        "queue_action_agreement": queue_action,
        "review": review,
    }, args.output_json)


#example:
#no human evaluation: python evaluate.py --result_json triage_results/triage_results_2026-11-07_01-04-58_300_Qwen3.5-9B_nothinking.json
#with human evaluation: python evaluate.py --result_json triage_results/triage_results_2026-11-07_01-04-58_300_Qwen3.5-9B_nothinking.json --human_review_csv --num_initial_eval 20
