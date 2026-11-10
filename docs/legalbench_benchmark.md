# LegalBench: the edited model against the raw model

Qwen3.5-0.8B as shipped (raw) against the same model with the formula-written edit of its label rows (edited),
one edit per task. Generated from `results/tables/rung3_all.csv` (script `analysis/run_rung3_all.py`).

- Score: balanced accuracy, the mean over labels of the share of that label's items answered correctly.
  Answering at random scores 1 / (number of labels).
- Raw: the model as shipped, its answer taken as the label whose first token it scores highest.
- Edited, 15 examples: the edit fitted on 15 labelled items drawn at random, scored on every other collected
  item; mean over 20 draws.
- Edited, all examples: two-fold, each item scored by an edit fitted on the other half of the sample (at most
  500 items per task).
- Word counts: the same formula fitted on word counts of the item instead of the model's state.
- Cut: share of prompts longer than 1,024 tokens, of which only the last 1,024 are read.

153 of the 162 LegalBench tasks. Not run: `citation_prediction_open`, `definition_extraction`, `rule_qa`, `sara_numeric`, `ssla_company_defendants`, `ssla_individual_defendants`, `ssla_plaintiff` (free-text answers), `successor_liability` (answers are sets of labels), `unfair_tos` (two labels share their first token).

## Summary

| Tasks | n | Raw | Edited, 15 examples | Word counts, 15 examples | Edited, all examples | Word counts, all examples |
|---|---|---|---|---|---|---|
| All tasks run | 153 | 0.524 | 0.683 | 0.607 | 0.774 | 0.726 |
| Two labels | 132 | 0.562 | 0.733 | 0.651 | 0.816 | 0.765 |
| Three to eight labels | 21 | 0.281 | 0.365 | 0.329 | 0.507 | 0.478 |
| CUAD contract clauses | 38 | 0.591 | 0.878 | 0.745 | 0.932 | 0.894 |
| Contract NLI | 14 | 0.598 | 0.845 | 0.738 | 0.903 | 0.856 |
| OPP-115 privacy policies | 9 | 0.628 | 0.777 | 0.647 | 0.870 | 0.803 |
| Learned Hands | 16 | 0.509 | 0.696 | 0.590 | 0.823 | 0.752 |
| MAUD merger agreements | 34 | 0.417 | 0.489 | 0.506 | 0.611 | 0.644 |
| Supply chain disclosures | 10 | 0.500 | 0.541 | 0.546 | 0.604 | 0.568 |
| Diversity jurisdiction | 6 | 0.615 | 0.691 | 0.503 | 0.883 | 0.552 |
| Other tasks | 26 | 0.487 | 0.575 | 0.512 | 0.662 | 0.573 |

## CUAD contract clauses

| Task | Labels | Items | Cut | Raw | Edited, 15 examples | Word counts, 15 examples | Edited, all examples | Word counts, all examples |
|---|---|---|---|---|---|---|---|---|
| `cuad_affiliate_license-licensee` | 2 | 198 | 6% | 0.778 | 0.866 | 0.729 | 0.889 | 0.889 |
| `cuad_affiliate_license-licensor` | 2 | 88 | 61% | 0.659 | 0.856 | 0.813 | 0.886 | 0.875 |
| `cuad_anti-assignment` | 2 | 1000 | 0% | 0.502 | 0.880 | 0.767 | 0.940 | 0.932 |
| `cuad_audit_rights` | 2 | 1000 | 0% | 0.572 | 0.932 | 0.687 | 0.978 | 0.929 |
| `cuad_cap_on_liability` | 2 | 1000 | 0% | 0.747 | 0.882 | 0.710 | 0.954 | 0.925 |
| `cuad_change_of_control` | 2 | 416 | 0% | 0.637 | 0.827 | 0.681 | 0.889 | 0.851 |
| `cuad_competitive_restriction_exception` | 2 | 220 | 0% | 0.505 | 0.682 | 0.570 | 0.786 | 0.764 |
| `cuad_covenant_not_to_sue` | 2 | 308 | 0% | 0.549 | 0.848 | 0.755 | 0.916 | 0.938 |
| `cuad_effective_date` | 2 | 236 | 0% | 0.513 | 0.969 | 0.801 | 0.975 | 0.958 |
| `cuad_exclusivity` | 2 | 762 | 0% | 0.727 | 0.813 | 0.634 | 0.852 | 0.822 |
| `cuad_expiration_date` | 2 | 876 | 0% | 0.635 | 0.921 | 0.812 | 0.948 | 0.936 |
| `cuad_governing_law` | 2 | 876 | 0% | 0.631 | 0.998 | 0.926 | 0.998 | 0.990 |
| `cuad_insurance` | 2 | 1000 | 0% | 0.567 | 0.929 | 0.701 | 0.978 | 0.944 |
| `cuad_ip_ownership_assignment` | 2 | 576 | 35% | 0.552 | 0.771 | 0.736 | 0.896 | 0.916 |
| `cuad_irrevocable_or_perpetual_license` | 2 | 280 | 100% | 0.718 | 0.887 | 0.829 | 0.950 | 0.954 |
| `cuad_joint_ip_ownership` | 2 | 192 | 100% | 0.635 | 0.849 | 0.744 | 0.927 | 0.875 |
| `cuad_license_grant` | 2 | 1000 | 2% | 0.565 | 0.862 | 0.777 | 0.908 | 0.887 |
| `cuad_liquidated_damages` | 2 | 220 | 1% | 0.509 | 0.917 | 0.732 | 0.950 | 0.850 |
| `cuad_minimum_commitment` | 2 | 772 | 0% | 0.526 | 0.781 | 0.650 | 0.892 | 0.864 |
| `cuad_most_favored_nation` | 2 | 64 | 0% | 0.500 | 0.845 | 0.752 | 0.891 | 0.812 |
| `cuad_no-solicit_of_customers` | 2 | 84 | 0% | 0.619 | 0.916 | 0.784 | 0.988 | 0.905 |
| `cuad_no-solicit_of_employees` | 2 | 142 | 1% | 0.528 | 0.958 | 0.840 | 0.993 | 0.944 |
| `cuad_non-compete` | 2 | 442 | 1% | 0.509 | 0.784 | 0.678 | 0.896 | 0.869 |
| `cuad_non-disparagement` | 2 | 100 | 0% | 0.550 | 0.922 | 0.692 | 0.930 | 0.910 |
| `cuad_non-transferable_license` | 2 | 542 | 2% | 0.517 | 0.850 | 0.769 | 0.918 | 0.890 |
| `cuad_notice_period_to_terminate_renewal` | 2 | 222 | 4% | 0.644 | 0.942 | 0.887 | 0.959 | 0.964 |
| `cuad_post-termination_services` | 2 | 808 | 2% | 0.500 | 0.795 | 0.654 | 0.920 | 0.812 |
| `cuad_price_restrictions` | 2 | 46 | 0% | 0.543 | 0.883 | 0.755 | 0.891 | 0.870 |
| `cuad_renewal_term` | 2 | 386 | 0% | 0.718 | 0.926 | 0.786 | 0.951 | 0.943 |
| `cuad_revenue-profit_sharing` | 2 | 774 | 0% | 0.518 | 0.936 | 0.709 | 0.972 | 0.898 |
| `cuad_rofr-rofo-rofn` | 2 | 690 | 1% | 0.510 | 0.826 | 0.652 | 0.936 | 0.896 |
| `cuad_source_code_escrow` | 2 | 118 | 1% | 0.619 | 0.917 | 0.709 | 0.966 | 0.873 |
| `cuad_termination_for_convenience` | 2 | 430 | 0% | 0.542 | 0.911 | 0.852 | 0.940 | 0.916 |
| `cuad_third_party_beneficiary` | 2 | 68 | 0% | 0.632 | 0.895 | 0.757 | 0.926 | 0.882 |
| `cuad_uncapped_liability` | 2 | 294 | 0% | 0.619 | 0.939 | 0.828 | 0.969 | 0.963 |
| `cuad_unlimited-all-you-can-eat-license` | 2 | 48 | 0% | 0.771 | 0.992 | 0.771 | 1.000 | 0.812 |
| `cuad_volume_restriction` | 2 | 322 | 0% | 0.522 | 0.806 | 0.666 | 0.925 | 0.835 |
| `cuad_warranty_duration` | 2 | 320 | 0% | 0.553 | 0.862 | 0.723 | 0.950 | 0.887 |

## Contract NLI

| Task | Labels | Items | Cut | Raw | Edited, 15 examples | Word counts, 15 examples | Edited, all examples | Word counts, all examples |
|---|---|---|---|---|---|---|---|---|
| `contract_nli_confidentiality_of_agreement` | 2 | 82 | 2% | 0.500 | 0.773 | 0.717 | 0.780 | 0.829 |
| `contract_nli_explicit_identification` | 2 | 109 | 0% | 0.550 | 0.687 | 0.581 | 0.863 | 0.708 |
| `contract_nli_inclusion_of_verbally_conveyed_information` | 2 | 139 | 1% | 0.630 | 0.850 | 0.729 | 0.864 | 0.862 |
| `contract_nli_limited_use` | 2 | 208 | 16% | 0.497 | 0.824 | 0.746 | 0.863 | 0.858 |
| `contract_nli_no_licensing` | 2 | 162 | 0% | 0.731 | 0.896 | 0.767 | 0.938 | 0.895 |
| `contract_nli_notice_on_compelled_disclosure` | 2 | 142 | 3% | 0.711 | 0.975 | 0.834 | 0.986 | 0.951 |
| `contract_nli_permissible_acquirement_of_similar_information` | 2 | 178 | 0% | 0.511 | 0.877 | 0.795 | 0.955 | 0.899 |
| `contract_nli_permissible_copy` | 2 | 87 | 0% | 0.499 | 0.707 | 0.590 | 0.810 | 0.777 |
| `contract_nli_permissible_development_of_similar_information` | 2 | 136 | 1% | 0.529 | 0.950 | 0.876 | 0.971 | 0.956 |
| `contract_nli_permissible_post-agreement_possession` | 2 | 111 | 11% | 0.577 | 0.727 | 0.612 | 0.922 | 0.762 |
| `contract_nli_return_of_confidential_information` | 2 | 66 | 0% | 0.609 | 0.879 | 0.763 | 0.911 | 0.881 |
| `contract_nli_sharing_with_employees` | 2 | 170 | 0% | 0.670 | 0.938 | 0.793 | 0.970 | 0.924 |
| `contract_nli_sharing_with_third-parties` | 2 | 180 | 0% | 0.620 | 0.821 | 0.733 | 0.842 | 0.769 |
| `contract_nli_survival_of_obligations` | 2 | 157 | 0% | 0.739 | 0.925 | 0.797 | 0.966 | 0.916 |

## OPP-115 privacy policies

| Task | Labels | Items | Cut | Raw | Edited, 15 examples | Word counts, 15 examples | Edited, all examples | Word counts, all examples |
|---|---|---|---|---|---|---|---|---|
| `opp115_data_retention` | 2 | 304 | 0% | 0.516 | 0.696 | 0.583 | 0.822 | 0.776 |
| `opp115_data_security` | 2 | 738 | 0% | 0.636 | 0.732 | 0.601 | 0.869 | 0.802 |
| `opp115_do_not_track` | 2 | 56 | 0% | 0.500 | 0.953 | 0.841 | 0.964 | 0.893 |
| `opp115_first_party_collection_use` | 2 | 1000 | 0% | 0.574 | 0.707 | 0.587 | 0.860 | 0.763 |
| `opp115_international_and_specific_audiences` | 2 | 694 | 0% | 0.702 | 0.862 | 0.620 | 0.938 | 0.852 |
| `opp115_policy_change` | 2 | 362 | 0% | 0.779 | 0.885 | 0.739 | 0.931 | 0.837 |
| `opp115_third_party_sharing_collection` | 2 | 1000 | 0% | 0.568 | 0.693 | 0.608 | 0.826 | 0.722 |
| `opp115_user_access,_edit_and_deletion` | 2 | 452 | 0% | 0.715 | 0.773 | 0.649 | 0.852 | 0.838 |
| `opp115_user_choice_control` | 2 | 1000 | 0% | 0.664 | 0.696 | 0.590 | 0.769 | 0.745 |

## Learned Hands

| Task | Labels | Items | Cut | Raw | Edited, 15 examples | Word counts, 15 examples | Edited, all examples | Word counts, all examples |
|---|---|---|---|---|---|---|---|---|
| `learned_hands_benefits` | 2 | 66 | 100% | 0.485 | 0.645 | 0.569 | 0.712 | 0.682 |
| `learned_hands_business` | 2 | 174 | 100% | 0.500 | 0.811 | 0.627 | 0.920 | 0.787 |
| `learned_hands_consumer` | 2 | 614 | 100% | 0.502 | 0.623 | 0.554 | 0.810 | 0.743 |
| `learned_hands_courts` | 2 | 192 | 100% | 0.500 | 0.622 | 0.542 | 0.740 | 0.661 |
| `learned_hands_crime` | 2 | 688 | 100% | 0.506 | 0.640 | 0.555 | 0.829 | 0.807 |
| `learned_hands_divorce` | 2 | 150 | 100% | 0.500 | 0.679 | 0.586 | 0.780 | 0.733 |
| `learned_hands_domestic_violence` | 2 | 174 | 100% | 0.500 | 0.695 | 0.591 | 0.787 | 0.707 |
| `learned_hands_education` | 2 | 56 | 100% | 0.500 | 0.672 | 0.608 | 0.732 | 0.643 |
| `learned_hands_employment` | 2 | 710 | 100% | 0.499 | 0.778 | 0.619 | 0.912 | 0.839 |
| `learned_hands_estates` | 2 | 178 | 100% | 0.494 | 0.751 | 0.628 | 0.876 | 0.820 |
| `learned_hands_family` | 2 | 1000 | 100% | 0.504 | 0.704 | 0.614 | 0.893 | 0.837 |
| `learned_hands_health` | 2 | 226 | 100% | 0.500 | 0.646 | 0.581 | 0.765 | 0.726 |
| `learned_hands_housing` | 2 | 1000 | 100% | 0.502 | 0.777 | 0.579 | 0.912 | 0.795 |
| `learned_hands_immigration` | 2 | 134 | 95% | 0.672 | 0.798 | 0.602 | 0.866 | 0.679 |
| `learned_hands_torts` | 2 | 432 | 100% | 0.481 | 0.577 | 0.547 | 0.748 | 0.720 |
| `learned_hands_traffic` | 2 | 556 | 100% | 0.498 | 0.716 | 0.635 | 0.886 | 0.856 |

## MAUD merger agreements

| Task | Labels | Items | Cut | Raw | Edited, 15 examples | Word counts, 15 examples | Edited, all examples | Word counts, all examples |
|---|---|---|---|---|---|---|---|---|
| `maud_ability_to_consummate_concept_is_subject_to_mae_carveouts` | 2 | 69 | 100% | 0.413 | 0.492 | 0.506 | 0.518 | 0.562 |
| `maud_accuracy_of_fundamental_target_rws_bringdown_standard` | 3 | 175 | 0% | 0.354 | 0.404 | 0.440 | 0.601 | 0.533 |
| `maud_accuracy_of_target_capitalization_rw_(outstanding_shares)_bringdown_standard_answer` | 4 | 148 | 94% | 0.222 | 0.281 | 0.328 | 0.368 | 0.378 |
| `maud_accuracy_of_target_general_rw_bringdown_timing_answer` | 2 | 181 | 0% | 0.604 | 0.659 | 0.573 | 0.791 | 0.693 |
| `maud_additional_matching_rights_period_for_modifications_(cor)` | 5 | 158 | 35% | 0.219 | 0.268 | 0.301 | 0.318 | 0.551 |
| `maud_application_of_buyer_consent_requirement_(negative_interim_covenant)` | 2 | 180 | 0% | 0.500 | 0.539 | 0.543 | 0.619 | 0.722 |
| `maud_buyer_consent_requirement_(ordinary_course)` | 2 | 181 | 3% | 0.531 | 0.604 | 0.677 | 0.704 | 0.827 |
| `maud_change_in_law__subject_to_disproportionate_impact_modifier` | 2 | 99 | 100% | 0.500 | 0.582 | 0.717 | 0.744 | 0.786 |
| `maud_changes_in_gaap_or_other_accounting_principles__subject_to_disproportionate_impact_modifier` | 2 | 98 | 100% | 0.500 | 0.553 | 0.652 | 0.686 | 0.719 |
| `maud_cor_permitted_in_response_to_intervening_event` | 2 | 100 | 27% | 0.432 | 0.593 | 0.597 | 0.540 | 0.664 |
| `maud_cor_permitted_with_board_fiduciary_determination_only` | 2 | 100 | 92% | 0.410 | 0.608 | 0.716 | 0.804 | 0.703 |
| `maud_cor_standard_(intervening_event)` | 6 | 84 | 19% | 0.176 | 0.349 | 0.391 | 0.561 | 0.609 |
| `maud_cor_standard_(superior_offer)` | 8 | 100 | 22% | 0.136 | 0.303 | 0.301 | 0.540 | 0.664 |
| `maud_definition_contains_knowledge_requirement_-_answer` | 4 | 124 | 2% | 0.270 | 0.305 | 0.396 | 0.339 | 0.435 |
| `maud_definition_includes_asset_deals` | 4 | 141 | 1% | 0.253 | 0.306 | 0.300 | 0.421 | 0.479 |
| `maud_definition_includes_stock_deals` | 4 | 142 | 1% | 0.250 | 0.260 | 0.266 | 0.264 | 0.338 |
| `maud_fiduciary_exception__board_determination_standard` | 8 | 179 | 12% | 0.132 | 0.363 | 0.391 | 0.749 | 0.741 |
| `maud_fiduciary_exception_board_determination_trigger_(no_shop)` | 2 | 179 | 50% | 0.390 | 0.682 | 0.751 | 0.893 | 0.988 |
| `maud_financial_point_of_view_is_the_sole_consideration` | 2 | 112 | 46% | 0.536 | 0.514 | 0.522 | 0.541 | 0.546 |
| `maud_fls_(mae)_standard` | 4 | 77 | 100% | 0.285 | 0.280 | 0.336 | 0.290 | 0.345 |
| `maud_general_economic_and_financial_conditions_subject_to_disproportionate_impact_modifier` | 2 | 98 | 100% | 0.500 | 0.544 | 0.677 | 0.619 | 0.774 |
| `maud_includes_consistent_with_past_practice` | 2 | 181 | 0% | 0.827 | 0.865 | 0.714 | 0.930 | 0.868 |
| `maud_initial_matching_rights_period_(cor)` | 5 | 158 | 46% | 0.209 | 0.251 | 0.262 | 0.319 | 0.510 |
| `maud_initial_matching_rights_period_(ftr)` | 5 | 132 | 31% | 0.317 | 0.232 | 0.283 | 0.368 | 0.541 |
| `maud_intervening_event_-_required_to_occur_after_signing_-_answer` | 2 | 124 | 16% | 0.585 | 0.524 | 0.556 | 0.607 | 0.616 |
| `maud_knowledge_definition` | 2 | 148 | 0% | 0.659 | 0.719 | 0.792 | 0.875 | 0.904 |
| `maud_liability_standard_for_no-shop_breach_by_target_non-do_representatives` | 2 | 134 | 6% | 0.499 | 0.562 | 0.556 | 0.549 | 0.640 |
| `maud_ordinary_course_efforts_standard` | 3 | 181 | 0% | 0.756 | 0.830 | 0.543 | 0.922 | 0.779 |
| `maud_pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier` | 2 | 98 | 100% | 0.500 | 0.577 | 0.549 | 0.615 | 0.620 |
| `maud_pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures` | 2 | 98 | 100% | 0.509 | 0.530 | 0.583 | 0.534 | 0.701 |
| `maud_relational_language_(mae)_applies_to` | 2 | 90 | 100% | 0.500 | 0.570 | 0.681 | 0.680 | 0.731 |
| `maud_specific_performance` | 2 | 178 | 0% | 0.535 | 0.573 | 0.532 | 0.920 | 0.689 |
| `maud_tail_period_length` | 4 | 179 | 0% | 0.242 | 0.330 | 0.278 | 0.875 | 0.688 |
| `maud_type_of_consideration` | 4 | 172 | 1% | 0.418 | 0.576 | 0.481 | 0.684 | 0.569 |

## Supply chain disclosures

| Task | Labels | Items | Cut | Raw | Edited, 15 examples | Word counts, 15 examples | Edited, all examples | Word counts, all examples |
|---|---|---|---|---|---|---|---|---|
| `supply_chain_disclosure_best_practice_accountability` | 2 | 379 | 16% | 0.500 | 0.535 | 0.530 | 0.578 | 0.504 |
| `supply_chain_disclosure_best_practice_audits` | 2 | 379 | 15% | 0.500 | 0.563 | 0.587 | 0.722 | 0.715 |
| `supply_chain_disclosure_best_practice_certification` | 2 | 378 | 15% | 0.500 | 0.548 | 0.522 | 0.603 | 0.548 |
| `supply_chain_disclosure_best_practice_training` | 2 | 379 | 16% | 0.500 | 0.550 | 0.574 | 0.619 | 0.649 |
| `supply_chain_disclosure_best_practice_verification` | 2 | 379 | 16% | 0.500 | 0.581 | 0.613 | 0.718 | 0.650 |
| `supply_chain_disclosure_disclosed_accountability` | 2 | 378 | 15% | 0.500 | 0.509 | 0.512 | 0.548 | 0.518 |
| `supply_chain_disclosure_disclosed_audits` | 2 | 379 | 16% | 0.500 | 0.533 | 0.528 | 0.537 | 0.499 |
| `supply_chain_disclosure_disclosed_certification` | 2 | 378 | 16% | 0.500 | 0.515 | 0.507 | 0.524 | 0.527 |
| `supply_chain_disclosure_disclosed_training` | 2 | 379 | 16% | 0.500 | 0.550 | 0.529 | 0.593 | 0.515 |
| `supply_chain_disclosure_disclosed_verification` | 2 | 379 | 16% | 0.500 | 0.525 | 0.561 | 0.593 | 0.558 |

## Diversity jurisdiction

| Task | Labels | Items | Cut | Raw | Edited, 15 examples | Word counts, 15 examples | Edited, all examples | Word counts, all examples |
|---|---|---|---|---|---|---|---|---|
| `diversity_1` | 2 | 300 | 0% | 0.673 | 0.692 | 0.509 | 0.943 | 0.506 |
| `diversity_2` | 2 | 300 | 0% | 0.615 | 0.718 | 0.495 | 0.947 | 0.556 |
| `diversity_3` | 2 | 300 | 0% | 0.660 | 0.726 | 0.504 | 0.828 | 0.520 |
| `diversity_4` | 2 | 300 | 0% | 0.675 | 0.782 | 0.505 | 0.953 | 0.615 |
| `diversity_5` | 2 | 300 | 0% | 0.570 | 0.672 | 0.502 | 0.852 | 0.533 |
| `diversity_6` | 2 | 300 | 0% | 0.495 | 0.553 | 0.500 | 0.774 | 0.580 |

## Other tasks

| Task | Labels | Items | Cut | Raw | Edited, 15 examples | Word counts, 15 examples | Edited, all examples | Word counts, all examples |
|---|---|---|---|---|---|---|---|---|
| `abercrombie` | 5 | 95 | 0% | 0.232 | 0.466 | 0.182 | 0.589 | 0.137 |
| `canada_tax_court_outcomes` | 3 | 244 | 0% | 0.344 | 0.680 | 0.535 | 0.900 | 0.641 |
| `citation_prediction_classification` | 2 | 108 | 0% | 0.491 | 0.447 | 0.444 | 0.324 | 0.241 |
| `consumer_contracts_qa` | 2 | 396 | 10% | 0.579 | 0.582 | 0.505 | 0.851 | 0.824 |
| `contract_qa` | 2 | 80 | 0% | 0.500 | 0.818 | 0.810 | 0.924 | 0.899 |
| `corporate_lobbying` | 2 | 490 | 91% | 0.497 | 0.544 | 0.577 | 0.640 | 0.729 |
| `definition_classification` | 2 | 1000 | 0% | 0.600 | 0.939 | 0.757 | 0.976 | 0.857 |
| `function_of_decision_section` | 7 | 367 | 0% | 0.193 | 0.268 | 0.209 | 0.440 | 0.417 |
| `hearsay` | 2 | 94 | 0% | 0.434 | 0.573 | 0.629 | 0.661 | 0.665 |
| `insurance_policy_interpretation` | 3 | 133 | 0% | 0.360 | 0.334 | 0.321 | 0.263 | 0.224 |
| `international_citizenship_questions` | 2 | 1000 | 0% | 0.500 | 0.522 | 0.572 | 0.604 | 0.631 |
| `jcrew_blocker` | 2 | 54 | 100% | 0.500 | 0.583 | 0.545 | 0.756 | 0.489 |
| `legal_reasoning_causality` | 2 | 55 | 100% | 0.532 | 0.703 | 0.776 | 0.847 | 0.926 |
| `nys_judicial_ethics` | 2 | 292 | 0% | 0.582 | 0.547 | 0.505 | 0.544 | 0.441 |
| `oral_argument_question_purpose` | 7 | 312 | 0% | 0.162 | 0.235 | 0.170 | 0.415 | 0.287 |
| `overruling` | 2 | 1000 | 0% | 0.666 | 0.873 | 0.662 | 0.937 | 0.897 |
| `personal_jurisdiction` | 2 | 50 | 0% | 0.588 | 0.495 | 0.506 | 0.511 | 0.518 |
| `privacy_policy_entailment` | 2 | 1000 | 100% | 0.600 | 0.643 | 0.575 | 0.836 | 0.713 |
| `privacy_policy_qa` | 2 | 1000 | 0% | 0.503 | 0.593 | 0.523 | 0.613 | 0.608 |
| `proa` | 2 | 95 | 0% | 0.531 | 0.813 | 0.648 | 0.874 | 0.737 |
| `sara_entailment` | 2 | 272 | 0% | 0.500 | 0.497 | 0.471 | 0.382 | 0.254 |
| `scalr` | 5 | 571 | 0% | 0.375 | 0.351 | 0.189 | 0.410 | 0.174 |
| `telemarketing_sales_rule` | 2 | 47 | 0% | 0.644 | 0.486 | 0.489 | 0.691 | 0.629 |
| `textualism_tool_dictionaries` | 2 | 107 | 100% | 0.575 | 0.525 | 0.537 | 0.611 | 0.500 |
| `textualism_tool_plain` | 2 | 165 | 58% | 0.529 | 0.589 | 0.594 | 0.681 | 0.712 |
| `ucc_v_common_law` | 2 | 94 | 0% | 0.648 | 0.836 | 0.579 | 0.931 | 0.738 |
