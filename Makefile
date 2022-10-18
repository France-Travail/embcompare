# Streamlit
demo:
	streamlit run embsuivi-gui/gui_compare.py
	
# Tests
test:
	pytest --cov=comp2vec tests/ --cov-report term-missing